"""
Meta Ads Multi-Agent System with Agno
Converted from n8n workflow with full token tracking

This system replicates the n8n workflow's 3-agent pipeline:
- Meta Main Agent: General Meta Ads operations (campaigns, adsets, ads, insights)
- Meta Creative Agent: Creative-specific operations (ad creatives, media analysis)
- Meta Report Agent: Report generation and comprehensive analysis
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.team.team import Team
from agno.tools.mcp import MCPTools
from agno.db.postgres import PostgresDb
from agno.guardrails import PIIDetectionGuardrail, PromptInjectionGuardrail
from rich.pretty import pprint
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# OpenRouter API Key (set via environment variable)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# MCP Server URL for Meta Ads
META_MCP_URL = "https://public-meta-mcp-production.up.railway.app/mcp"

# Database for metrics storage (optional - can use SQLite for local development)
DB_URL = os.getenv("AGNO_DB_URL", "sqlite:///meta_ads_metrics.db")


# ============================================================================
# Output Schemas (from n8n workflow response format)
# ============================================================================

class MetricItem(BaseModel):
    """Individual metric with optional change indicator"""
    label: str
    value: str
    change: Optional[str] = None
    trend: Optional[str] = None  # "up" or "down"
    period: Optional[str] = None


class HighlightCard(BaseModel):
    """Highlight card for top performer or key insight"""
    title: str
    subtitle: Optional[str] = None
    badge: Optional[Dict[str, str]] = None  # {"text": "...", "variant": "success|warning|error|info"}
    icon: Optional[str] = None  # play, image, document, chat, sparkles, trending-up, trending-down
    metrics: List[MetricItem]


class CreativeContent(BaseModel):
    """Creative content with media and metrics"""
    heading: Optional[str] = None
    asset_type: str  # "image" or "video"
    asset_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    text: Optional[str] = None
    analysis: Optional[str] = None
    metrics: List[MetricItem]


class ActionItem(BaseModel):
    """Actionable recommendation"""
    number: int
    text: str
    buttonText: str  # "Review", "Apply", "Create"


class FollowUpItem(BaseModel):
    """Follow-up suggestion"""
    text: str


class MetaAdsResponse(BaseModel):
    """Structured response format for Meta Ads analysis"""
    highlight: Optional[HighlightCard] = None
    creative_metrics: Optional[Dict[str, Any]] = None
    table: Optional[Dict[str, Any]] = None
    chart: Optional[Dict[str, Any]] = None
    actions: Optional[Dict[str, Any]] = None
    follow_ups: Optional[Dict[str, List[FollowUpItem]]] = None


# ============================================================================
# MCP Tool Configurations (matching n8n workflow tool selections)
# ============================================================================

# Tools for Meta Main Agent (general operations + CRUD)
META_MAIN_TOOLS = [
    "get_details_of_ad_account",
    "get_campaigns_by_adaccount",
    "get_campaign_by_id",
    "get_adsets_by_adaccount",
    "get_adsets_by_campaign",
    "get_adset_by_id",
    "get_adsets_by_ids",
    "get_ads_by_adaccount",
    "get_ads_by_adset",
    "get_ad_by_id",
    "get_insights",
    "get_campaign_insights",
    "get_adset_insights",
    "get_ad_insights",
    "get_adaccount_insights",
    "get_activities_by_adaccount",
    "get_activities_by_adset",
    "fetch_pagination_url",
    "search_pages_by_name",
    "get_account_pages",
    "update_campaign_tool",
    "update_adset_tool",
    "update_ad_tool",
    "update_ad_creative_tool",
    "get_ads_by_campaign",
    "create_campaign_tool",
    "create_adset_tool",
    "create_ad_tool",
    "upload_ad_image_tool",
    "upload_ad_video_tool",
    "create_ad_creative_tool",
    "get_user_selected_ad_account",
]

# Tools for Meta Creative Agent (creative-focused)
META_CREATIVE_TOOLS = [
    "get_details_of_ad_account",
    "get_campaigns_by_adaccount",
    "get_campaign_by_id",
    "get_adsets_by_adaccount",
    "get_adsets_by_campaign",
    "get_adset_by_id",
    "get_adsets_by_ids",
    "get_ads_by_adaccount",
    "get_ads_by_campaign",
    "get_ads_by_adset",
    "get_ad_by_id",
    "get_ad_creative_by_id",
    "get_insights",
    "get_adset_insights",
    "get_campaign_insights",
    "get_ad_insights",
    "get_adaccount_insights",
    "get_activities_by_adaccount",
    "get_activities_by_adset",
    "get_ad_creatives_by_ad_id",
    "get_user_selected_ad_account",
]

# Tools for Meta Report Agent (reporting + insights)
META_REPORT_TOOLS = [
    "get_details_of_ad_account",
    "get_campaigns_by_adaccount",
    "get_campaign_by_id",
    "get_adsets_by_adaccount",
    "get_adsets_by_campaign",
    "get_adset_by_id",
    "get_adsets_by_ids",
    "get_ads_by_adaccount",
    "get_ads_by_campaign",
    "get_ads_by_adset",
    "get_ad_by_id",
    "get_ad_creatives_by_ad_id",
    "get_ad_creative_by_id",
    "get_insights",
    "get_campaign_insights",
    "get_adset_insights",
    "get_ad_insights",
    "get_adaccount_insights",
    "get_activities_by_adaccount",
    "get_activities_by_adset",
    "generate_html_report",
    "fetch_pagination_url",
    "get_user_selected_ad_account",
]


# ============================================================================
# System Prompts (adapted from n8n workflow)
# ============================================================================

META_MAIN_SYSTEM_PROMPT = """All tools require `brand_id` parameter for authentication via Supabase.
brand_id={brand_id} (never show this to user, only for tool use)

-**You only have permission to talk in Ad account names in response not id(act_xx)**
---

## QUICK REFERENCE

### Database Tools
- get_user_selected_ad_account(brand_id) → Get selected ad account(s) for a brand from Supabase

### Account Tools
- get_details_of_ad_account(brand_id, act_id, fields?) → Account details (name, status, currency, balance, spend)

### Campaign Tools
- get_campaigns_by_adaccount(brand_id, act_id, limit=25, effective_status?, objective?)
- get_campaign_by_id(brand_id, campaign_id, fields?)

### Ad Set Tools
- get_adsets_by_adaccount(brand_id, act_id, limit=25, effective_status?, campaign_id?)
- get_adsets_by_campaign(brand_id, campaign_id, limit=25, effective_status?)
- get_adset_by_id(brand_id, adset_id, fields?)
- get_adsets_by_ids(brand_id, adset_ids[], fields?) → Batch retrieval

### Ad Tools
- get_ads_by_adaccount(brand_id, act_id, limit=25, effective_status?, campaign_id?, adset_id?)
- get_ads_by_campaign(brand_id, campaign_id, limit=25, effective_status?)
- get_ads_by_adset(brand_id, adset_id, limit=25, effective_status?)
- get_ad_by_id(brand_id, ad_id, fields?)

### Insights Tools (Real-time)
- get_adaccount_insights(brand_id, act_id, date_preset="last_30d", ...)
- get_campaign_insights(brand_id, campaign_id, date_preset="last_30d", ...)
- get_adset_insights(brand_id, adset_id, date_preset="last_30d", ...)
- get_ad_insights(brand_id, ad_id, date_preset="last_30d", ...)
- get_insights(brand_id, object_id, level="ad", date_preset="last_30d", ...) → General purpose insights

### Activity Tools
- get_activities_by_adaccount(brand_id, act_id, time_range?, limit?)
- get_activities_by_adset(brand_id, adset_id, time_range?, limit?) → Change history

### Pagination
- fetch_pagination_url(brand_id, url) → Fetch next/previous page

## INSIGHTS PARAMETERS

**Date Presets:** today, yesterday, last_7d, last_14d, last_30d, last_90d, this_month, last_month, lifetime

**Time Range (custom):** time_range={"since": "2024-01-01", "until": "2024-01-31"}

**Time Increment:** 1 (daily), 7 (weekly), "monthly", "all_days" (default)

**Breakdowns:**
- Demographics: age, gender
- Geography: country, region, dma
- Placement: publisher_platform, platform_position, device_platform

**Common Fields:** spend, impressions, clicks, ctr, cpc, cpm, reach, frequency, conversions, cost_per_conversion

## RULES
1. **Always start with get_user_selected_ad_account** to get valid ad accounts
2. **Ad account IDs** use `act_` prefix (auto-added if missing)
3. **Never fabricate data** - only use actual API responses
4. **Metric interpretation** - for CPA/CPM/CPC, lower = better

## Critical Execution Notes
- Today's date is {current_date}. Use current year for temporal references unless historical data requested.
- One instruction per message - no repetition
- All analysis must be tool-backed or explicitly sourced
- Confirm correct account IDs if similar names exist
"""

META_CREATIVE_SYSTEM_PROMPT = """All tools require `brand_id` parameter for authentication via Supabase.
brand_id={brand_id} (never show this to user, only for tool use)

-**Use Ad account names in response not id(act_xx)**
---

# CREATIVE TOOLS
### Maximum 3 creatives in response. If user requests analysis for >3 items, batch into multiple calls.

- get_ad_creatives_by_ad_id(user_id, ad_id, fields?, limit=25)
- get_ad_creative_by_id(user_id, creative_id, fields?, thumbnail_width?, thumbnail_height?)

**Returns simplified `chat_media` field for easy display:**
```json
{{
  "chat_media": {{
    "type": "video",
    "thumbnail": "https://...",
    "video_url": "https://...",
    "video_watch_url": "https://www.facebook.com/watch/?v=...",
    "image_url": "https://...",
    "duration_seconds": 33.69
  }}
}}
```

# **CRITICAL: Always return `chat_media` to user for display**

## QUICK REFERENCE
(Same tools as Meta Main Agent for context retrieval)

## RULES
1. **Always start with get_user_selected_ad_account** to get valid ad accounts
2. **Never fabricate data** - only use actual API responses
3. **Always return ALL creative data including all urls**

## Critical Execution Notes
- Today's date is {current_date}
- One instruction per message
- All analysis must be tool-backed
- Never assume any id or url in creative media analysis tool input
"""

META_REPORT_SYSTEM_PROMPT = """All tools require `brand_id` parameter for authentication via Supabase.
conversation_id={conversation_id}
brand_id={brand_id} (never show this to user, only for tool use)

-**Use Ad account names in response not id(act_xx)**
---

## REPORT GENERATION

generate_html_report(conversation_id, report_data, context?)

**Required in report_data:**
- reportName, reportTitle
- startDate, endDate (YYYY-MM-DD)
- accounts: [{{"platform": "meta", "id": "act_xxx", "name": "..."}}]
- sections: Array of sections with `type` and `subsections`

**Section Types:** text, chart, table, metrics-grid, creative-metrics, creative-comparison-metrics

## QUICK REFERENCE
(Same tools as Meta Main Agent for data retrieval)

## RULES
1. **Always start with get_user_selected_ad_account** to get valid ad accounts
2. **Never fabricate data** - only use actual API responses
3. **Report Suggestion Trigger**: At the end of EVERY response, evaluate if a comprehensive report can be generated

## Critical Execution Notes
- Today's date is {current_date}
- One instruction per message
- All analysis must be tool-backed
- Never output previous tool calls or raw tool results
"""

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Marketing AI Assistant specializing in AI-assisted Meta Ads performance analysis and optimization.

You orchestrate and route queries to specialized sub-agents:
- **Meta Main Agent**: General Meta Ads operations (campaigns, ad sets, ads, insights, CRUD operations)
- **Meta Creative Agent**: Creative-specific operations (ad creatives, media analysis, thumbnails)
- **Meta Report Agent**: Report generation and comprehensive analysis

## Platform & Account Routing

You receive account information in the context:
- Use account names in responses, IDs only in tool calls
- Ignore accounts from conversation history if new accounts are provided

## Sub-agent Routing Logic

1. **Meta Creative Agent** - When query is about:
   - Creative performance
   - Ad creative analysis
   - Video/image assets
   - Thumbnail generation

2. **Meta Report Agent** - When user explicitly asks for:
   - Reports
   - Comprehensive analysis
   - Executive summaries
   - Exportable insights

3. **Meta Main Agent** - For all other Meta-related questions:
   - Campaign management
   - Budget optimization
   - Audience insights
   - General performance metrics

## Response Format

Use structured JSON cards for responses:
- `highlight` - Top performer or key insight
- `performance-table` - Comparison data
- `creative-metrics` - Creative analysis with media
- `actions` - Recommendations
- `follow-ups` - Related questions (always include)

## Critical Rules
- Never fabricate data
- All analysis must be tool-backed
- Use account names in responses, IDs in tool calls
- Today's date is {current_date}
"""


# ============================================================================
# Token Tracking Utilities (using Agno's built-in metrics)
# ============================================================================

def get_metrics_from_response(run_response) -> Dict[str, Any]:
    """
    Extract metrics using Agno's built-in tracking.

    Agno automatically tracks token usage, timing, and model info.
    This function extracts and formats those metrics.
    """
    if not run_response or not run_response.metrics:
        return {
            "total": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "time_to_first_token": None,
            "response_time": None,
            "model": "unknown"
        }

    metrics = run_response.metrics
    input_tokens = getattr(metrics, 'input_tokens', 0) or 0
    output_tokens = getattr(metrics, 'output_tokens', 0) or 0

    return {
        "total": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "time_to_first_token": getattr(metrics, 'time_to_first_token', None),
        "response_time": getattr(metrics, 'response_time', None),
        "model": getattr(metrics, 'model', 'unknown'),
    }


def print_metrics_summary(metrics: Dict[str, Any]):
    """Print a formatted summary of metrics"""
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)

    total = metrics.get("total", {})
    print(f"\nTotal Tokens: {total.get('total_tokens', 0):,}")
    print(f"  - Input:  {total.get('input_tokens', 0):,}")
    print(f"  - Output: {total.get('output_tokens', 0):,}")

    if metrics.get("response_time"):
        print(f"\nResponse Time: {metrics['response_time']:.2f}s")
    if metrics.get("time_to_first_token"):
        print(f"Time to First Token: {metrics['time_to_first_token']:.2f}s")
    if metrics.get("model"):
        print(f"Model: {metrics['model']}")

    print("=" * 60)


# Legacy compatibility - keep TokenTracker for backward compatibility
class TokenTracker:
    """
    Legacy token tracker for backward compatibility.
    New code should use get_metrics_from_response() and team.get_session_metrics().
    """

    def __init__(self):
        self.usage_log: List[Dict[str, Any]] = []

    def record_from_response(self, run_response, agent_name: str = "agent"):
        """Record metrics from a run response"""
        metrics = get_metrics_from_response(run_response)
        metrics["agent"] = agent_name
        metrics["timestamp"] = datetime.now().isoformat()
        self.usage_log.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary of all recorded metrics"""
        total_input = sum(m.get("total", {}).get("input_tokens", 0) for m in self.usage_log)
        total_output = sum(m.get("total", {}).get("output_tokens", 0) for m in self.usage_log)

        return {
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output
            },
            "call_count": len(self.usage_log)
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()
        print_metrics_summary(summary)


# ============================================================================
# Agent Factory Functions
# ============================================================================

def create_meta_main_agent(brand_id: str, mcp_tools: MCPTools) -> Agent:
    """Create the Meta Main Agent with general Meta Ads tools"""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Agent(
        name="Meta Main Agent",
        model=OpenRouter(
            id="google/gemini-2.5-flash",
            api_key=OPENROUTER_API_KEY
        ),
        role="General Meta Ads operations including campaigns, ad sets, ads, and insights",
        tools=[mcp_tools],
        instructions=META_MAIN_SYSTEM_PROMPT.format(
            brand_id=brand_id,
            current_date=current_date
        ),
        markdown=True,
        show_tool_calls=True,
    )


def create_meta_creative_agent(brand_id: str, mcp_tools: MCPTools) -> Agent:
    """Create the Meta Creative Agent for creative-specific operations"""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Agent(
        name="Meta Creative Agent",
        model=OpenRouter(
            id="anthropic/claude-sonnet-4.5",
            api_key=OPENROUTER_API_KEY
        ),
        role="Creative analysis including ad creatives, video/image assets, and media analysis",
        tools=[mcp_tools],
        instructions=META_CREATIVE_SYSTEM_PROMPT.format(
            brand_id=brand_id,
            current_date=current_date
        ),
        markdown=True,
        show_tool_calls=True,
    )


def create_meta_report_agent(brand_id: str, conversation_id: str, mcp_tools: MCPTools) -> Agent:
    """Create the Meta Report Agent for report generation"""
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Agent(
        name="Meta Report Agent",
        model=OpenRouter(
            id="anthropic/claude-sonnet-4.5",
            api_key=OPENROUTER_API_KEY
        ),
        role="Report generation and comprehensive analysis",
        tools=[mcp_tools],
        instructions=META_REPORT_SYSTEM_PROMPT.format(
            brand_id=brand_id,
            conversation_id=conversation_id,
            current_date=current_date
        ),
        markdown=True,
        show_tool_calls=True,
    )


async def create_meta_ads_team_async(
    brand_id: str,
    conversation_id: str,
    session_id: Optional[str] = None,
    db_url: Optional[str] = None
) -> tuple[Team, List[MCPTools]]:
    """
    Create the Meta Ads Team with all three agents (async version).

    Returns the team and a list of MCP connections that need to be closed.

    This replicates the n8n workflow structure:
    - Orchestrator (AI Agent2) routes to specialized agents
    - Meta Main Agent handles general operations
    - Meta Creative Agent handles creative analysis
    - Meta Report Agent handles report generation

    Features enabled:
    - PostgreSQL-backed session persistence
    - Automatic conversation history
    - Per-conversation memory
    - Shared team state
    - PII detection and injection protection guardrails (log-only)
    """
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Configure PostgreSQL for persistence (if available)
    db = None
    effective_db_url = db_url or os.getenv("AGNO_DB_URL")
    if effective_db_url:
        db = PostgresDb(
            db_url=effective_db_url,
            session_table="meta_ads_sessions"
        )

    # Create and connect MCP tools for each agent
    mcp_main = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_MAIN_TOOLS
    )
    mcp_creative = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_CREATIVE_TOOLS
    )
    mcp_report = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_REPORT_TOOLS
    )

    # Connect all MCP tools
    await mcp_main.connect()
    await mcp_creative.connect()
    await mcp_report.connect()

    mcp_connections = [mcp_main, mcp_creative, mcp_report]

    # Create the three specialized agents with connected MCP tools
    meta_main = create_meta_main_agent(brand_id, mcp_main)
    meta_creative = create_meta_creative_agent(brand_id, mcp_creative)
    meta_report = create_meta_report_agent(brand_id, conversation_id, mcp_report)

    # Create the orchestrating team with full Agno features
    team = Team(
        name="Meta Ads Team",
        model=OpenRouter(
            id="anthropic/claude-3.5-haiku",  # Using haiku for orchestration (faster, cheaper)
            api_key=OPENROUTER_API_KEY
        ),
        members=[meta_main, meta_creative, meta_report],
        instructions=[
            ORCHESTRATOR_SYSTEM_PROMPT.format(current_date=current_date),
            "Route creative-related queries to Meta Creative Agent",
            "Route report requests to Meta Report Agent",
            "Route all other Meta queries to Meta Main Agent",
        ],
        # Database persistence
        db=db,
        session_id=session_id or conversation_id,

        # Auto-include conversation history from database
        add_history_to_context=True,

        # Per-conversation memory
        enable_agentic_memory=True,

        # Shared state for team members
        enable_agentic_state=True,

        # Guardrails (log-only mode for PII and injection detection)
        pre_hooks=[
            PIIDetectionGuardrail(action="log"),
            PromptInjectionGuardrail(action="log"),
        ],

        # Response handling
        markdown=True,
        show_members_responses=True,
        store_member_responses=True,  # Required for metrics tracking
    )

    return team, mcp_connections


async def close_mcp_connections(connections: List[MCPTools]):
    """Close all MCP connections"""
    for conn in connections:
        try:
            await conn.close()
        except Exception as e:
            print(f"Error closing MCP connection: {e}")


# Legacy sync function for backward compatibility
def create_meta_ads_team(
    brand_id: str,
    conversation_id: str,
    session_id: Optional[str] = None,
    db_url: Optional[str] = None
) -> Team:
    """
    Create the Meta Ads Team (sync wrapper - NOT RECOMMENDED).

    Note: This creates a team but MCP tools won't be connected.
    For proper MCP tool usage, use create_meta_ads_team_async() instead.
    """
    import warnings
    warnings.warn(
        "create_meta_ads_team() is deprecated. Use create_meta_ads_team_async() "
        "for proper MCP tool connection handling.",
        DeprecationWarning
    )

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Configure PostgreSQL for persistence (if available)
    db = None
    effective_db_url = db_url or os.getenv("AGNO_DB_URL")
    if effective_db_url:
        db = PostgresDb(
            db_url=effective_db_url,
            session_table="meta_ads_sessions"
        )

    # Create MCP tools (not connected - will be connected by Agno internally if supported)
    mcp_main = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_MAIN_TOOLS
    )
    mcp_creative = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_CREATIVE_TOOLS
    )
    mcp_report = MCPTools(
        transport="streamable-http",
        url=META_MCP_URL,
        include_tools=META_REPORT_TOOLS
    )

    # Create agents
    meta_main = create_meta_main_agent(brand_id, mcp_main)
    meta_creative = create_meta_creative_agent(brand_id, mcp_creative)
    meta_report = create_meta_report_agent(brand_id, conversation_id, mcp_report)

    # Create the orchestrating team with full Agno features
    team = Team(
        name="Meta Ads Team",
        model=OpenRouter(
            id="anthropic/claude-3.5-haiku",
            api_key=OPENROUTER_API_KEY
        ),
        members=[meta_main, meta_creative, meta_report],
        instructions=[
            ORCHESTRATOR_SYSTEM_PROMPT.format(current_date=current_date),
            "Route creative-related queries to Meta Creative Agent",
            "Route report requests to Meta Report Agent",
            "Route all other Meta queries to Meta Main Agent",
        ],
        db=db,
        session_id=session_id or conversation_id,
        add_history_to_context=True,
        enable_agentic_memory=True,
        enable_agentic_state=True,
        pre_hooks=[
            PIIDetectionGuardrail(action="log"),
            PromptInjectionGuardrail(action="log"),
        ],
        markdown=True,
        show_members_responses=True,
        store_member_responses=True,
    )

    return team


# ============================================================================
# Main Execution Functions
# ============================================================================

async def run_meta_ads_query(
    user_message: str,
    brand_id: str,
    conversation_id: str,
    ad_accounts: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run a query through the Meta Ads multi-agent system.

    Agno automatically handles:
    - Conversation history (via PostgreSQL if configured)
    - Token tracking (via built-in metrics)
    - Session state persistence

    Args:
        user_message: The user's query
        brand_id: Brand ID for Supabase authentication
        conversation_id: Conversation ID for context (used as session_id)
        ad_accounts: Dict of ad accounts by platform

    Returns:
        Dict containing response, metrics, and session_metrics
    """
    # Create the team with connected MCP tools
    team, mcp_connections = await create_meta_ads_team_async(
        brand_id=brand_id,
        conversation_id=conversation_id
    )

    try:
        # Build context message (simplified - no more manual history fetch)
        full_message = user_message
        if ad_accounts:
            full_message = f"Selected Ad Accounts: {ad_accounts}\n\nUser Query: {user_message}"

        # Run the team
        run_response = await team.arun(full_message)

        # Get metrics using Agno's built-in tracking
        metrics = get_metrics_from_response(run_response)

        # Get session metrics (aggregated across all calls in this conversation)
        session_metrics = None
        try:
            session_metrics = team.get_session_metrics(session_id=conversation_id)
        except Exception:
            pass  # Session metrics may not be available without DB

        return {
            "response": run_response,
            "content": run_response.content if run_response else None,
            "metrics": metrics,
            "session_metrics": session_metrics,
        }
    finally:
        # Always close MCP connections
        await close_mcp_connections(mcp_connections)


def run_meta_ads_query_sync(
    user_message: str,
    brand_id: str,
    conversation_id: str,
    ad_accounts: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Synchronous version of run_meta_ads_query.

    Note: This uses asyncio.run internally. For better performance in
    async contexts, use run_meta_ads_query() instead.

    Args:
        user_message: The user's query
        brand_id: Brand ID for Supabase authentication
        conversation_id: Conversation ID for context (used as session_id)
        ad_accounts: Dict of ad accounts by platform

    Returns:
        Dict containing response, metrics, and session_metrics
    """
    import asyncio
    return asyncio.run(run_meta_ads_query(
        user_message=user_message,
        brand_id=brand_id,
        conversation_id=conversation_id,
        ad_accounts=ad_accounts
    ))


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    result = run_meta_ads_query_sync(
        user_message="What are the top performing campaigns in the last 7 days?",
        brand_id="example-brand-id",
        conversation_id="example-conversation-id",
        ad_accounts={
            "meta": [
                {"id": "act_123456789", "name": "My Ad Account"}
            ]
        },
    )

    # Print response
    print("\n" + "=" * 60)
    print("RESPONSE")
    print("=" * 60)
    print(result["content"])

    # Print token usage (using Agno's built-in metrics)
    print_metrics_summary(result["metrics"])

    # Print session metrics if available
    if result.get("session_metrics"):
        print("\n" + "=" * 60)
        print("SESSION METRICS (Aggregated)")
        print("=" * 60)
        pprint(result["session_metrics"])
