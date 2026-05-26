use std::path::PathBuf;

use codex_app_server_protocol::CommandExecutionStatus;
use codex_app_server_protocol::McpToolCallStatus;
use codex_app_server_protocol::ServerNotification;
use codex_app_server_protocol::ThreadItem;
use codex_app_server_protocol::TurnStatus;
use codex_core::config::Config;
use codex_protocol::protocol::SessionConfiguredEvent;
use serde_json::json;
use serde_json::Value as JsonValue;

use crate::event_processor::CodexStatus;
use crate::event_processor::EventProcessor;
use crate::event_processor::handle_last_message;

/// Emits Claude-compatible stream-json NDJSON on stdout.
///
/// Protocol: bidirectional NDJSON matching Claude Code's `claude -p
/// --input-format=stream-json --output-format=stream-json`.
///
/// Key differences from Codex JSONL:
/// - Aggregates items into per-message frames
/// - Emits tool_use blocks inside assistant messages
/// - Emits tool_results as separate user messages
/// - Emits thinking blocks inside assistant messages
pub(crate) struct StreamJsonProcessor {
    /// Buffered content blocks for the current assistant message.
    content_blocks: Vec<JsonValue>,
    /// Monotonic message counter for generating `msg_N` IDs.
    msg_counter: u64,
    /// Pending tool results to flush before the next assistant message.
    pending_tool_results: Vec<JsonValue>,
    /// Maps raw item ID to stable tool_use_id.
    active_tools: std::collections::HashMap<String, String>,
    /// Model name from config.
    model: Option<String>,
    /// Session ID from SessionConfiguredEvent.
    session_id: Option<String>,
    /// Path to write the final agent message.
    last_message_path: Option<PathBuf>,
    /// Last agent message text for last_message_path.
    final_message: Option<String>,
    /// Whether we've emitted the system/init frame yet.
    init_emitted: bool,
    /// Accumulated token usage from ThreadTokenUsageUpdated.
    input_tokens: i64,
    output_tokens: i64,
    cached_input_tokens: i64,
}

impl StreamJsonProcessor {
    pub fn new(last_message_path: Option<PathBuf>, model: Option<String>) -> Self {
        Self {
            content_blocks: Vec::new(),
            msg_counter: 0,
            pending_tool_results: Vec::new(),
            active_tools: std::collections::HashMap::new(),
            model,
            session_id: None,
            last_message_path,
            final_message: None,
            init_emitted: false,
            input_tokens: 0,
            output_tokens: 0,
            cached_input_tokens: 0,
        }
    }

    fn next_msg_id(&mut self) -> String {
        self.msg_counter += 1;
        format!("msg_{}", self.msg_counter)
    }

    fn next_tool_use_id(&mut self, raw_item_id: &str) -> String {
        format!("tu_{raw_item_id}")
    }

    /// Emit a single JSON line to stdout.
    #[allow(clippy::print_stdout)]
    fn emit(&self, frame: JsonValue) {
        println!(
            "{}",
            serde_json::to_string(&frame).unwrap_or_else(|err| {
                json!({"type": "error", "message": format!("serialize error: {err}")}).to_string()
            })
        );
    }

    fn emit_init(&mut self) {
        if self.init_emitted {
            return;
        }
        self.init_emitted = true;
        self.emit(json!({
            "type": "system",
            "subtype": "init",
            "session_id": self.session_id.as_deref().unwrap_or(""),
            "model": self.model.as_deref().unwrap_or(""),
        }));
    }

    /// Flush any buffered content_blocks as an assistant message frame.
    fn flush_assistant_message(&mut self) {
        if self.content_blocks.is_empty() {
            return;
        }
        let blocks = std::mem::take(&mut self.content_blocks);
        let msg_id = self.next_msg_id();
        self.emit(json!({
            "type": "assistant",
            "message": {
                "id": msg_id,
                "content": blocks,
            }
        }));
    }

    /// Flush pending tool results as a user message frame.
    fn flush_tool_results(&mut self) {
        if self.pending_tool_results.is_empty() {
            return;
        }
        let results = std::mem::take(&mut self.pending_tool_results);
        self.emit(json!({
            "type": "user",
            "message": {
                "content": results,
            }
        }));
    }

    /// Emit the result frame at end of turn.
    fn emit_result(&self, subtype: &str, error_msg: Option<&str>) {
        let mut frame = json!({
            "type": "result",
            "subtype": subtype,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_input_tokens": self.cached_input_tokens,
                "cache_creation_input_tokens": 0,
            },
        });
        if let Some(msg) = error_msg {
            frame["error"] = json!({"message": msg});
        }
        self.emit(frame);
    }
}

impl EventProcessor for StreamJsonProcessor {
    fn print_config_summary(
        &mut self,
        _config: &Config,
        _prompt: &str,
        session_configured: &SessionConfiguredEvent,
    ) {
        self.session_id = Some(session_configured.thread_id.to_string());
        self.emit_init();
    }

    fn process_server_notification(&mut self, notification: ServerNotification) -> CodexStatus {
        match notification {
            ServerNotification::TurnStarted(_) => {
                self.emit_init();
                CodexStatus::Running
            }
            ServerNotification::ItemStarted(notification) => {
                self.handle_item_started(notification.item);
                CodexStatus::Running
            }
            ServerNotification::ItemCompleted(notification) => {
                self.handle_item_completed(notification.item);
                CodexStatus::Running
            }
            ServerNotification::ThreadTokenUsageUpdated(notification) => {
                let usage = &notification.token_usage.total;
                self.input_tokens = usage.input_tokens;
                self.output_tokens = usage.output_tokens;
                self.cached_input_tokens = usage.cached_input_tokens;
                CodexStatus::Running
            }
            ServerNotification::TurnCompleted(notification) => {
                // Flush any remaining assistant content.
                self.flush_assistant_message();
                self.flush_tool_results();

                match notification.turn.status {
                    TurnStatus::Completed => {
                        if let Some(msg) = self.final_message_from_items(&notification.turn.items) {
                            self.final_message = Some(msg);
                        }
                        self.emit_result("success", None);
                        CodexStatus::InitiateShutdown
                    }
                    TurnStatus::Failed => {
                        let error_msg = notification
                            .turn
                            .error
                            .as_ref()
                            .map(|e| e.message.as_str())
                            .unwrap_or("turn failed");
                        self.emit_result("error", Some(error_msg));
                        CodexStatus::InitiateShutdown
                    }
                    TurnStatus::Interrupted => {
                        self.emit_result("error", Some("interrupted"));
                        CodexStatus::InitiateShutdown
                    }
                    TurnStatus::InProgress => {
                        CodexStatus::Running
                    }
                }
            }
            ServerNotification::Error(notification) => {
                // Do NOT emit a `result` frame here — `result` is terminal in
                // the Claude Code protocol, but Error notifications can arrive
                // mid-turn.  Two cases:
                //
                //   will_retry=true: transient error (e.g. WebSocket reconnect).
                //     The turn continues; no terminal frame needed.
                //
                //   will_retry=false: fatal error.  A TurnCompleted(Failed)
                //     will follow and emit the proper terminal `result/error`.
                //
                // Match the human/JSONL processors: log to stderr and keep
                // running.
                let retry_tag = if notification.will_retry {
                    " (will retry)"
                } else {
                    ""
                };
                eprintln!(
                    "error: {}{}",
                    notification.error.message, retry_tag
                );
                CodexStatus::Running
            }
            ServerNotification::ConfigWarning(notification) => {
                eprintln!("warning: {}", notification.summary);
                CodexStatus::Running
            }
            _ => CodexStatus::Running,
        }
    }

    fn process_warning(&mut self, message: String) -> CodexStatus {
        eprintln!("warning: {message}");
        CodexStatus::Running
    }

    fn print_final_output(&mut self) {
        if let Some(path) = &self.last_message_path {
            handle_last_message(self.final_message.as_deref(), path);
        }
    }
}

impl StreamJsonProcessor {
    fn handle_item_started(&mut self, item: ThreadItem) {
        match item {
            ThreadItem::AgentMessage { .. } | ThreadItem::Reasoning { .. } => {
                // Text/thinking content is buffered; actual content added on completed.
            }
            ThreadItem::CommandExecution {
                id,
                command,
                cwd,
                ..
            } => {
                // Flush any pending text/thinking first.
                self.flush_assistant_message();
                self.flush_tool_results();

                let tool_use_id = self.next_tool_use_id(&id);
                // Emit tool_use in its own assistant message.
                let msg_id = self.next_msg_id();
                self.emit(json!({
                    "type": "assistant",
                    "message": {
                        "id": msg_id,
                        "content": [{
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "Bash",
                            "input": {
                                "command": command,
                                "workdir": cwd.display().to_string(),
                            }
                        }]
                    }
                }));
                self.active_tools.insert(id, tool_use_id);
            }
            ThreadItem::McpToolCall {
                id,
                server,
                tool,
                arguments,
                ..
            } => {
                self.flush_assistant_message();
                self.flush_tool_results();

                let tool_use_id = self.next_tool_use_id(&id);
                let tool_name = format!("mcp__{server}__{tool}");
                let msg_id = self.next_msg_id();
                self.emit(json!({
                    "type": "assistant",
                    "message": {
                        "id": msg_id,
                        "content": [{
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_name,
                            "input": arguments,
                        }]
                    }
                }));
                self.active_tools.insert(id, tool_use_id);
            }
            ThreadItem::FileChange { id, .. } => {
                self.flush_assistant_message();
                self.flush_tool_results();

                let tool_use_id = self.next_tool_use_id(&id);
                let msg_id = self.next_msg_id();
                self.emit(json!({
                    "type": "assistant",
                    "message": {
                        "id": msg_id,
                        "content": [{
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "file_change",
                            "input": {}
                        }]
                    }
                }));
                self.active_tools.insert(id, tool_use_id);
            }
            ThreadItem::WebSearch { id, query, .. } => {
                self.flush_assistant_message();
                self.flush_tool_results();

                let tool_use_id = self.next_tool_use_id(&id);
                let msg_id = self.next_msg_id();
                self.emit(json!({
                    "type": "assistant",
                    "message": {
                        "id": msg_id,
                        "content": [{
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": "web_search",
                            "input": {"query": query}
                        }]
                    }
                }));
                self.active_tools.insert(id, tool_use_id);
            }
            _ => {}
        }
    }

    fn handle_item_completed(&mut self, item: ThreadItem) {
        match item {
            ThreadItem::AgentMessage { text, .. } => {
                if !text.is_empty() {
                    self.final_message = Some(text.clone());
                    self.content_blocks.push(json!({
                        "type": "text",
                        "text": text,
                    }));
                }
            }
            ThreadItem::Reasoning { summary, content, .. } => {
                let thinking_text = if !summary.is_empty() {
                    summary.join("\n")
                } else if !content.is_empty() {
                    content.join("\n")
                } else {
                    return;
                };
                self.content_blocks.push(json!({
                    "type": "thinking",
                    "thinking": thinking_text,
                }));
            }
            ThreadItem::CommandExecution {
                id,
                aggregated_output,
                exit_code,
                status,
                ..
            } => {
                if let Some(tool_use_id) = self.active_tools.remove(&id) {
                    let output = aggregated_output.unwrap_or_default();
                    let is_error = match status {
                        CommandExecutionStatus::Failed => true,
                        _ => exit_code.map(|c| c != 0).unwrap_or(false),
                    };
                    self.pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output,
                        "is_error": is_error,
                    }));
                }
            }
            ThreadItem::McpToolCall {
                id,
                result,
                error,
                status,
                ..
            } => {
                if let Some(tool_use_id) = self.active_tools.remove(&id) {
                    let (content, is_error) = match status {
                        McpToolCallStatus::Failed => {
                            let msg = error
                                .as_ref()
                                .map(|e| e.message.clone())
                                .unwrap_or_else(|| "tool call failed".to_string());
                            (msg, true)
                        }
                        _ => {
                            let content = result
                                .as_ref()
                                .map(|r| {
                                    serde_json::to_string(&r.content)
                                        .unwrap_or_else(|_| "null".to_string())
                                })
                                .unwrap_or_default();
                            (content, false)
                        }
                    };
                    self.pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                        "is_error": is_error,
                    }));
                }
            }
            ThreadItem::FileChange { id, changes, .. } => {
                if let Some(tool_use_id) = self.active_tools.remove(&id) {
                    let diff_text: String = changes
                        .iter()
                        .map(|c| format!("{}:\n{}", c.path, c.diff))
                        .collect::<Vec<_>>()
                        .join("\n");
                    self.pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": diff_text,
                        "is_error": false,
                    }));
                }
            }
            ThreadItem::WebSearch { id, .. } => {
                if let Some(tool_use_id) = self.active_tools.remove(&id) {
                    self.pending_tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "",
                        "is_error": false,
                    }));
                }
            }
            _ => {}
        }
    }

    fn final_message_from_items(&self, items: &[ThreadItem]) -> Option<String> {
        items.iter().rev().find_map(|item| match item {
            ThreadItem::AgentMessage { text, .. } if !text.is_empty() => Some(text.clone()),
            _ => None,
        })
    }
}
