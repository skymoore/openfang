//! Tool classification for Programmatic Tool Calling.
//!
//! Determines which tools should remain as direct JSON tool calls
//! (with full schemas in the LLM prompt) and which should be callable
//! only via `execute_code` (schemas removed, compact function signatures
//! shown instead).
//!
//! Default: ALL tools are PTC-eligible. There are no tools that technically
//! must be direct — the classification exists for future extensibility.

use openfang_types::tool::ToolDefinition;

/// Whether a tool is called directly by the LLM or via `execute_code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtcMode {
    /// Tool is called directly by the LLM (full JSON schema in prompt).
    Direct,
    /// Tool is callable only via `execute_code` (schema removed from prompt).
    Ptc,
}

/// Classify a single tool.
///
/// Currently all tools are PTC-eligible. This function exists as an
/// extension point for future per-tool overrides.
pub fn classify_tool(tool: &ToolDefinition) -> PtcMode {
    // The execute_code tool itself must be direct (it IS the PTC entry point)
    if tool.name == "execute_code" {
        return PtcMode::Direct;
    }

    // Everything else is callable via code
    PtcMode::Ptc
}

/// Split tools into direct and PTC sets.
///
/// Returns `(direct_tools, ptc_tools)`.
pub fn classify_tools(tools: &[ToolDefinition]) -> (Vec<ToolDefinition>, Vec<ToolDefinition>) {
    let mut direct = Vec::new();
    let mut ptc = Vec::new();

    for tool in tools {
        match classify_tool(tool) {
            PtcMode::Direct => direct.push(tool.clone()),
            PtcMode::Ptc => ptc.push(tool.clone()),
        }
    }

    (direct, ptc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("Test tool {name}"),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    #[test]
    fn test_all_tools_are_ptc_by_default() {
        let tools = vec![
            make_tool("file_read"),
            make_tool("web_search"),
            make_tool("shell_exec"),
            make_tool("agent_send"),
            make_tool("mcp_github_list"),
        ];

        let (direct, ptc) = classify_tools(&tools);
        assert!(direct.is_empty());
        assert_eq!(ptc.len(), 5);
    }

    #[test]
    fn test_execute_code_is_always_direct() {
        let tool = make_tool("execute_code");
        assert_eq!(classify_tool(&tool), PtcMode::Direct);
    }
}
