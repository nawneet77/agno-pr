# Meta Ads Multi-Agent System (Agno)

A Python-based multi-agent system for Meta Ads management, converted from n8n workflow to Agno framework with full token tracking.

## Architecture

This system replicates the n8n workflow's 3-agent pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                    Meta Ads Team                            │
│                  (Orchestrator Agent)                       │
│              Model: claude-3.5-haiku                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Meta Main   │  │ Meta        │  │ Meta Report │         │
│  │ Agent       │  │ Creative    │  │ Agent       │         │
│  │             │  │ Agent       │  │             │         │
│  │ gemini-2.5  │  │ claude-     │  │ claude-     │         │
│  │ -flash      │  │ sonnet-4.5  │  │ sonnet-4.5  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│        │                │                │                  │
│        └────────────────┴────────────────┘                  │
│                         │                                   │
│              ┌──────────┴──────────┐                       │
│              │    MCP Tools        │                       │
│              │  (Meta Ads API)     │                       │
│              └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Agents

1. **Meta Main Agent** (gemini-2.5-flash via OpenRouter)
   - General Meta Ads operations
   - Campaign, ad set, ad management
   - Insights and analytics
   - CRUD operations

2. **Meta Creative Agent** (claude-sonnet-4.5 via OpenRouter)
   - Creative-specific operations
   - Ad creative analysis
   - Video/image asset handling
   - Media thumbnails

3. **Meta Report Agent** (claude-sonnet-4.5 via OpenRouter)
   - Report generation
   - Comprehensive analysis
   - HTML report creation

## Features

- **Full n8n Workflow Compatibility**: Same webhook interface, same request/response format
- **Token Tracking**: Built-in Agno metrics for input/output tokens, timing, and model info
- **MCP Tools Integration**: Uses the same Meta MCP server as n8n
- **Streaming Support**: Real-time SSE streaming responses
- **PostgreSQL Session Persistence**: Automatic conversation history via Agno's database
- **Per-Conversation Memory**: Agents remember context within each conversation
- **Shared Team State**: Team members access shared state (selected accounts, date range)
- **Guardrails**: PII detection and prompt injection protection (log-only mode)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
# Required
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Required for full Agno features (sessions, memory, state)
export AGNO_DB_URL="postgresql+psycopg://user:pass@localhost:5432/db"
```

**Note:** Supabase is no longer required. Agno handles conversation history, session persistence, and memory automatically via PostgreSQL.

## Usage

### Run the Server

```bash
# Start the FastAPI server
python meta_ads_server.py

# Or with uvicorn directly
uvicorn meta_ads_server:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Main Webhook (matches n8n)
```bash
POST /webhook/open-ronin-mcp
Content-Type: application/json

{
  "conversation_id": "uuid",
  "user_message": "What are my top campaigns?",
  "user_id": "uuid",
  "brand_id": "uuid",
  "ad_accounts": {
    "meta": [{"id": "act_xxx", "name": "My Account"}]
  },
  "date": "2025-12-13:2026-01-12"
}
```

#### Streaming Webhook
```bash
POST /webhook/open-ronin-mcp/stream
# Same body as above, returns SSE stream
```

#### Individual Agent Endpoints
```bash
POST /agents/meta-main
POST /agents/meta-creative
POST /agents/meta-report
```

### Programmatic Usage

```python
from meta_ads_agents import (
    create_meta_ads_team,
    run_meta_ads_query_sync,
    TokenTracker
)

# Create tracker for token metrics
tracker = TokenTracker()

# Run a query
result = run_meta_ads_query_sync(
    user_message="What are the top performing campaigns?",
    brand_id="your-brand-id",
    conversation_id="your-conversation-id",
    ad_accounts={
        "meta": [{"id": "act_123", "name": "My Account"}]
    },
    tracker=tracker
)

# Access response
print(result["content"])

# Print token usage
tracker.print_summary()
```

### Async Usage

```python
import asyncio
from meta_ads_agents import run_meta_ads_query, TokenTracker

async def main():
    tracker = TokenTracker()

    result = await run_meta_ads_query(
        user_message="Analyze my ad creatives",
        brand_id="your-brand-id",
        conversation_id="your-conversation-id",
        tracker=tracker
    )

    print(result["content"])
    print(result["metrics"])

asyncio.run(main())
```

## Token Tracking

The system uses Agno's built-in metrics tracking:

```python
# Metrics structure (per request)
{
    "total": {
        "input_tokens": 1500,
        "output_tokens": 800,
        "total_tokens": 2300
    },
    "time_to_first_token": 0.45,
    "response_time": 2.3,
    "model": "anthropic/claude-3.5-haiku"
}

# Session metrics (aggregated across conversation)
# Available via team.get_session_metrics(session_id)
```

## MCP Tools Available

### Meta Main Agent Tools
- `get_campaigns_by_adaccount`, `get_campaign_by_id`
- `get_adsets_by_adaccount`, `get_adsets_by_campaign`, `get_adset_by_id`
- `get_ads_by_adaccount`, `get_ads_by_adset`, `get_ad_by_id`
- `get_*_insights` (account, campaign, adset, ad level)
- `create_campaign_tool`, `create_adset_tool`, `create_ad_tool`
- `update_campaign_tool`, `update_adset_tool`, `update_ad_tool`
- And more...

### Meta Creative Agent Tools
- All read tools from Main Agent
- `get_ad_creative_by_id`
- `get_ad_creatives_by_ad_id`

### Meta Report Agent Tools
- All read tools from Main Agent
- `generate_html_report`

## Comparison with n8n Workflow

| Feature | n8n Workflow | Agno System |
|---------|--------------|-------------|
| Orchestrator | AI Agent (claude-3.5-haiku) | Team (claude-3.5-haiku) |
| Meta Main | Agent Tool (gemini-2.5-flash) | Agent (gemini-2.5-flash) |
| Meta Creative | Agent Tool (claude-sonnet-4.5) | Agent (claude-sonnet-4.5) |
| Meta Report | Agent Tool (claude-sonnet-4.5) | Agent (claude-sonnet-4.5) |
| MCP Tools | MCP Client nodes | MCPTools class |
| Token Tracking | Not built-in | Agno built-in metrics |
| Streaming | Via webhook response mode | SSE endpoint |
| Session Persistence | Manual Supabase fetch | Automatic via PostgreSQL |
| Memory | Not available | Per-conversation memory |
| Guardrails | Not available | PII/Injection detection |
| State Management | Not available | Shared team state |

## Agno Features Utilized

This implementation leverages the following Agno features:

1. **PostgresDb** - Session persistence and conversation history
2. **PIIDetectionGuardrail** - Logs detected PII in requests
3. **PromptInjectionGuardrail** - Logs potential injection attempts
4. **enable_agentic_memory** - Per-conversation memory
5. **enable_agentic_state** - Shared state between team members
6. **add_history_to_context** - Automatic conversation history injection
7. **Built-in metrics** - Token tracking without custom code

## License

MIT
