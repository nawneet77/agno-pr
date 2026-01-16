"""
FastAPI Server for Meta Ads Multi-Agent System

This server provides a webhook-compatible interface matching the n8n workflow:
- POST /open-ronin-mcp - Main endpoint for Meta Ads queries
- Streaming response support
- Full token tracking and metrics

Environment Variables:
- OPENROUTER_API_KEY: Your OpenRouter API key
- SUPABASE_URL: Supabase URL for chat history (optional)
- SUPABASE_KEY: Supabase API key (optional)
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.team.team import Team
from agno.tools.mcp import MCPTools

# Import from our agents module
from meta_ads_agents import (
    create_meta_ads_team_async,
    close_mcp_connections,
    run_meta_ads_query,
    get_metrics_from_response,
    META_MCP_URL,
    OPENROUTER_API_KEY,
    create_meta_main_agent,
    create_meta_creative_agent,
    create_meta_report_agent,
    META_MAIN_TOOLS,
    META_CREATIVE_TOOLS,
    META_REPORT_TOOLS,
)


# ============================================================================
# Request/Response Models (matching n8n webhook format)
# ============================================================================

class AdAccount(BaseModel):
    """Ad account model"""
    id: str
    name: str


class AdAccounts(BaseModel):
    """Container for ad accounts by platform"""
    meta: Optional[List[AdAccount]] = None
    tiktok: Optional[List[AdAccount]] = None
    google_analytics: Optional[List[AdAccount]] = None
    google_ads: Optional[List[AdAccount]] = None
    shopify: Optional[List[AdAccount]] = None


class FileAttachment(BaseModel):
    """File attachment model"""
    url: str
    name: Optional[str] = None
    type: Optional[str] = None


class WebhookRequest(BaseModel):
    """
    Request model matching n8n webhook body format

    Example:
    {
        "conversation_id": "uuid",
        "user_message": "What are my top campaigns?",
        "user_id": "uuid",
        "brand_id": "uuid",
        "files": null,
        "ad_accounts": {
            "meta": [{"id": "act_xxx", "name": "My Account"}],
            "tiktok": null
        },
        "date": "2025-12-13:2026-01-12"
    }
    """
    conversation_id: str
    user_message: str
    user_id: str
    brand_id: str
    files: Optional[List[FileAttachment]] = None
    ad_accounts: Optional[AdAccounts] = None
    date: Optional[str] = None  # Format: "YYYY-MM-DD:YYYY-MM-DD"


class TokenUsage(BaseModel):
    """Token usage metrics"""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class MetricsResponse(BaseModel):
    """Metrics response model"""
    total: TokenUsage
    by_agent: Dict[str, TokenUsage]
    by_model: Dict[str, TokenUsage]
    call_count: int


class WebhookResponse(BaseModel):
    """Response model for non-streaming responses"""
    content: str
    conversation_id: str
    metrics: Optional[MetricsResponse] = None


# ============================================================================
# FastAPI Application
# ============================================================================
# NOTE: Conversation history is now handled automatically by Agno's
# PostgreSQL-backed sessions. No manual Supabase fetch needed.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("Meta Ads Multi-Agent Server starting...")
    print(f"OpenRouter API Key configured: {'Yes' if OPENROUTER_API_KEY else 'No'}")
    print(f"MCP Server URL: {META_MCP_URL}")
    yield
    print("Meta Ads Multi-Agent Server shutting down...")


app = FastAPI(
    title="Meta Ads Multi-Agent API",
    description="Agno-based multi-agent system for Meta Ads management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openrouter_configured": bool(OPENROUTER_API_KEY)
    }


@app.post("/webhook/open-ronin-mcp")
async def meta_ads_webhook(request: WebhookRequest):
    """
    Main webhook endpoint matching n8n workflow.

    Accepts the same payload format as the n8n webhook node.
    Conversation history is handled automatically by Agno's PostgreSQL sessions.
    """
    try:
        # Convert ad_accounts to dict format
        ad_accounts_dict = None
        if request.ad_accounts:
            ad_accounts_dict = {
                "meta": [acc.dict() for acc in request.ad_accounts.meta] if request.ad_accounts.meta else None,
                "tiktok": [acc.dict() for acc in request.ad_accounts.tiktok] if request.ad_accounts.tiktok else None,
                "google_analytics": [acc.dict() for acc in request.ad_accounts.google_analytics] if request.ad_accounts.google_analytics else None,
                "google_ads": [acc.dict() for acc in request.ad_accounts.google_ads] if request.ad_accounts.google_ads else None,
                "shopify": [acc.dict() for acc in request.ad_accounts.shopify] if request.ad_accounts.shopify else None,
            }

        # Build the full message with date context
        full_message = request.user_message
        if request.date:
            try:
                start_date, end_date = request.date.split(":")
                full_message += f"\nDate range selected: {start_date} to {end_date}"
            except ValueError:
                pass

        # Run query using Agno (handles history automatically now)
        result = await run_meta_ads_query(
            user_message=full_message,
            brand_id=request.brand_id,
            conversation_id=request.conversation_id,
            ad_accounts=ad_accounts_dict,
        )

        return JSONResponse(content={
            "content": result["content"] or "",
            "conversation_id": request.conversation_id,
            "metrics": result["metrics"],
            "session_metrics": result.get("session_metrics"),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/open-ronin-mcp/stream")
async def meta_ads_webhook_streaming(request: WebhookRequest):
    """
    Streaming version of the webhook endpoint.

    Returns Server-Sent Events (SSE) for real-time streaming.
    Conversation history is handled automatically by Agno's PostgreSQL sessions.
    """
    async def generate():
        mcp_connections = []
        try:
            # Convert ad_accounts
            ad_accounts_dict = None
            if request.ad_accounts:
                ad_accounts_dict = {
                    "meta": [acc.dict() for acc in request.ad_accounts.meta] if request.ad_accounts.meta else None,
                    "tiktok": [acc.dict() for acc in request.ad_accounts.tiktok] if request.ad_accounts.tiktok else None,
                    "google_analytics": [acc.dict() for acc in request.ad_accounts.google_analytics] if request.ad_accounts.google_analytics else None,
                    "google_ads": [acc.dict() for acc in request.ad_accounts.google_ads] if request.ad_accounts.google_ads else None,
                    "shopify": [acc.dict() for acc in request.ad_accounts.shopify] if request.ad_accounts.shopify else None,
                }

            # Build message with date context
            full_message = request.user_message
            if request.date:
                try:
                    start_date, end_date = request.date.split(":")
                    full_message += f"\nDate range selected: {start_date} to {end_date}"
                except ValueError:
                    pass

            # Add ad accounts context
            if ad_accounts_dict:
                full_message = f"Selected Ad Accounts: {json.dumps(ad_accounts_dict)}\n\nUser Query: {full_message}"

            # Create team with connected MCP tools
            team, mcp_connections = await create_meta_ads_team_async(
                brand_id=request.brand_id,
                conversation_id=request.conversation_id
            )

            # Stream the response
            full_content = ""
            final_response = None
            async for chunk in team.arun_stream(full_message):
                if hasattr(chunk, 'content') and chunk.content:
                    full_content += chunk.content
                    yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
                final_response = chunk

            # Send final metrics using Agno's built-in tracking
            metrics = get_metrics_from_response(final_response)
            yield f"data: {json.dumps({'type': 'metrics', 'data': metrics})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': request.conversation_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        finally:
            # Always close MCP connections
            await close_mcp_connections(mcp_connections)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/metrics")
async def get_global_metrics():
    """
    Get global metrics (placeholder - would need persistent storage)
    """
    return {
        "message": "Metrics are returned per-request. See /webhook/open-ronin-mcp response.",
        "note": "For persistent metrics, configure a database in the team setup."
    }


# ============================================================================
# Individual Agent Endpoints (for direct access)
# ============================================================================

@app.post("/agents/meta-main")
async def run_meta_main_agent(
    message: str = Query(..., description="User message"),
    brand_id: str = Query(..., description="Brand ID for authentication"),
    ad_accounts: Optional[str] = Query(None, description="JSON string of ad accounts")
):
    """Direct access to Meta Main Agent"""
    mcp_tools = None
    try:
        # Create and connect MCP tools
        mcp_tools = MCPTools(
            transport="streamable-http",
            url=META_MCP_URL,
            include_tools=META_MAIN_TOOLS
        )
        await mcp_tools.connect()

        agent = create_meta_main_agent(brand_id, mcp_tools)

        # Parse ad_accounts if provided
        ad_accounts_dict = None
        if ad_accounts:
            try:
                ad_accounts_dict = json.loads(ad_accounts)
            except json.JSONDecodeError:
                pass  # Invalid JSON, ignore

        context = f"Ad Accounts: {json.dumps(ad_accounts_dict)}" if ad_accounts_dict else ""
        full_message = f"{context}\n\n{message}" if context else message

        response = await agent.arun(full_message)
        metrics = get_metrics_from_response(response)

        return {
            "content": response.content if response else "",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if mcp_tools:
            try:
                await mcp_tools.close()
            except Exception:
                pass


@app.post("/agents/meta-creative")
async def run_meta_creative_agent(
    message: str = Query(..., description="User message"),
    brand_id: str = Query(..., description="Brand ID for authentication"),
    ad_accounts: Optional[str] = Query(None, description="JSON string of ad accounts")
):
    """Direct access to Meta Creative Agent"""
    mcp_tools = None
    try:
        # Create and connect MCP tools
        mcp_tools = MCPTools(
            transport="streamable-http",
            url=META_MCP_URL,
            include_tools=META_CREATIVE_TOOLS
        )
        await mcp_tools.connect()

        agent = create_meta_creative_agent(brand_id, mcp_tools)

        # Parse ad_accounts if provided
        ad_accounts_dict = None
        if ad_accounts:
            try:
                ad_accounts_dict = json.loads(ad_accounts)
            except json.JSONDecodeError:
                pass  # Invalid JSON, ignore

        context = f"Ad Accounts: {json.dumps(ad_accounts_dict)}" if ad_accounts_dict else ""
        full_message = f"{context}\n\n{message}" if context else message

        response = await agent.arun(full_message)
        metrics = get_metrics_from_response(response)

        return {
            "content": response.content if response else "",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if mcp_tools:
            try:
                await mcp_tools.close()
            except Exception:
                pass


@app.post("/agents/meta-report")
async def run_meta_report_agent(
    message: str = Query(..., description="User message"),
    brand_id: str = Query(..., description="Brand ID for authentication"),
    conversation_id: str = Query(..., description="Conversation ID for context"),
    ad_accounts: Optional[str] = Query(None, description="JSON string of ad accounts")
):
    """Direct access to Meta Report Agent"""
    mcp_tools = None
    try:
        # Create and connect MCP tools
        mcp_tools = MCPTools(
            transport="streamable-http",
            url=META_MCP_URL,
            include_tools=META_REPORT_TOOLS
        )
        await mcp_tools.connect()

        agent = create_meta_report_agent(brand_id, conversation_id, mcp_tools)

        # Parse ad_accounts if provided
        ad_accounts_dict = None
        if ad_accounts:
            try:
                ad_accounts_dict = json.loads(ad_accounts)
            except json.JSONDecodeError:
                pass  # Invalid JSON, ignore

        context = f"Ad Accounts: {json.dumps(ad_accounts_dict)}" if ad_accounts_dict else ""
        full_message = f"{context}\n\n{message}" if context else message

        response = await agent.arun(full_message)
        metrics = get_metrics_from_response(response)

        return {
            "content": response.content if response else "",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if mcp_tools:
            try:
                await mcp_tools.close()
            except Exception:
                pass


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "meta_ads_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
