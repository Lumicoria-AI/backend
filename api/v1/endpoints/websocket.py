"""
WebSocket endpoint for real-time notification delivery.

This module provides:
- WebSocket connection endpoint at /ws/notifications/{user_id}
- Authentication for WebSocket connections
- Real-time notification broadcasting
- Connection heartbeat/ping-pong
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from typing import Optional
import json
import structlog
import asyncio

from backend.services.notification_service import connection_manager
from backend.services.presence_service import presence_broker
from backend.core.security import verify_token

logger = structlog.get_logger()

router = APIRouter()


async def authenticate_websocket(
    websocket: WebSocket,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Authenticate a WebSocket connection using JWT token.
    
    Returns user_id if authenticated, None otherwise.
    """
    if not token:
        return None
    
    try:
        payload = await verify_token(token)
        if payload:
            # security.verify_token returns a dict with 'user_id'
            return payload.get("user_id") or payload.get("sub")
    except Exception as e:
        logger.warning("websocket_auth_failed", error=str(e))
    
    return None


@router.websocket("/notifications/{user_id}")
async def websocket_notifications(
    websocket: WebSocket,
    user_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time notification delivery.
    
    Connect: ws://host:port/ws/notifications/{user_id}?token=JWT_TOKEN
    
    Message types sent to client:
    - notification: New notification created
    - notification_read: A notification was marked as read
    - all_notifications_read: All notifications marked as read
    - notification_deleted: A notification was deleted
    - ping: Heartbeat ping (client should respond with pong)
    
    Message types accepted from client:
    - pong: Heartbeat response
    - subscribe: Subscribe to additional notification types
    - unsubscribe: Unsubscribe from notification types
    """
    # Authenticate the connection
    authenticated_user_id = await authenticate_websocket(websocket, token)
    
    # Allow connection if token authenticates to the requested user_id
    # or if no token is provided (for development/testing)
    if token and authenticated_user_id != user_id:
        logger.warning(
            "websocket_unauthorized",
            requested_user_id=user_id,
            authenticated_user_id=authenticated_user_id
        )
        await websocket.close(code=4001, reason="Unauthorized")
        return
    
    # Accept and register connection
    await connection_manager.connect(websocket, user_id)
    
    # Send initial connection acknowledgment
    await websocket.send_json({
        "type": "connected",
        "data": {
            "user_id": user_id,
            "message": "WebSocket connection established"
        }
    })
    
    # Start heartbeat task
    heartbeat_task = asyncio.create_task(
        send_heartbeat(websocket, user_id)
    )
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "pong":
                    # Heartbeat response - connection is alive
                    logger.debug("websocket_pong_received", user_id=user_id)
                    
                elif message_type == "subscribe":
                    # Client wants to subscribe to specific notification types
                    notification_types = message.get("data", {}).get("types", [])
                    logger.info(
                        "websocket_subscribe",
                        user_id=user_id,
                        types=notification_types
                    )
                    await websocket.send_json({
                        "type": "subscribed",
                        "data": {"types": notification_types}
                    })
                    
                elif message_type == "unsubscribe":
                    # Client wants to unsubscribe from specific notification types
                    notification_types = message.get("data", {}).get("types", [])
                    logger.info(
                        "websocket_unsubscribe",
                        user_id=user_id,
                        types=notification_types
                    )
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "data": {"types": notification_types}
                    })
                    
                elif message_type == "get_status":
                    # Client requests connection status
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "connected": True,
                            "user_id": user_id
                        }
                    })
                    
                else:
                    logger.debug(
                        "websocket_unknown_message",
                        user_id=user_id,
                        message_type=message_type
                    )
                    
            except json.JSONDecodeError:
                logger.warning("websocket_invalid_json", user_id=user_id)
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                })
                
    except WebSocketDisconnect:
        logger.info("websocket_disconnect", user_id=user_id)
    except Exception as e:
        logger.error("websocket_error", user_id=user_id, error=str(e))
    finally:
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        
        # Disconnect and cleanup
        connection_manager.disconnect(websocket, user_id)


async def send_heartbeat(websocket: WebSocket, user_id: str):
    """
    Send periodic heartbeat pings to keep connection alive.
    
    Sends a ping every 30 seconds. If client doesn't respond with pong,
    the connection may be considered stale.
    """
    while True:
        try:
            await asyncio.sleep(30)  # Ping every 30 seconds
            await websocket.send_json({
                "type": "ping",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })
            logger.debug("websocket_ping_sent", user_id=user_id)
        except Exception as e:
            logger.error("websocket_heartbeat_error", user_id=user_id, error=str(e))
            break


@router.get("/status")
async def get_websocket_status():
    """
    Get WebSocket server status and connected user count.
    """
    connected_users = len(connection_manager.active_connections)
    total_connections = sum(
        len(conns) for conns in connection_manager.active_connections.values()
    )
    
    return {
        "status": "operational",
        "connected_users": connected_users,
        "total_connections": total_connections
    }


@router.websocket("/presence")
async def websocket_presence(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
):
    """Realtime presence + typing.

    Connect: ws://host:port/api/v1/ws/presence?token=JWT&organization_id=<oid>

    Server → client message types:
      - connected                { user_id, organization_id, online_user_ids }
      - presence.update          { user_id, online, last_seen? }
      - typing.start             { room, user_id, at }
      - typing.stop              { room, user_id, at }
      - pong                     ping response

    Client → server message types:
      - subscribe                { room: "project:<id>" | "team:<id>" | "chat:<id>" | "task:<id>" }
      - unsubscribe              { room }
      - typing                   { room, typing: true|false }
      - ping                     heartbeat
    """
    authenticated_user_id = await authenticate_websocket(websocket, token)
    if not authenticated_user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    if not organization_id:
        await websocket.close(code=4400, reason="organization_id required")
        return

    await websocket.accept()
    await presence_broker.mark_online(websocket, user_id=authenticated_user_id, organization_id=organization_id)

    try:
        await websocket.send_json({
            "type": "connected",
            "data": {
                "user_id": authenticated_user_id,
                "organization_id": organization_id,
                "online_user_ids": presence_broker.online_users_for_org(organization_id),
            },
        })
    except Exception:
        pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = (msg or {}).get("type")
            data = (msg or {}).get("data") or {}

            if mtype == "ping":
                try:
                    await websocket.send_json({"type": "pong"})
                except Exception:
                    pass
            elif mtype == "subscribe":
                room = str(data.get("room") or "").strip()
                if room:
                    await presence_broker.subscribe(websocket, user_id=authenticated_user_id, room=room)
            elif mtype == "unsubscribe":
                room = str(data.get("room") or "").strip()
                if room:
                    await presence_broker.unsubscribe(websocket, user_id=authenticated_user_id, room=room)
            elif mtype == "typing":
                room = str(data.get("room") or "").strip()
                typing_flag = bool(data.get("typing", True))
                if room:
                    await presence_broker.broadcast_typing(
                        room=room, user_id=authenticated_user_id, typing=typing_flag,
                    )
            # Unknown types are ignored.
    except WebSocketDisconnect:
        logger.info("presence_websocket_disconnect", user_id=authenticated_user_id, organization_id=organization_id)
    except Exception as e:
        logger.exception("presence_websocket_error", error=str(e))
    finally:
        await presence_broker.mark_offline(websocket, user_id=authenticated_user_id)


@router.get("/presence/snapshot")
async def presence_snapshot(organization_id: str):
    """Snapshot of who is currently online in the org."""
    return {
        "organization_id": organization_id,
        "online_user_ids": presence_broker.online_users_for_org(organization_id),
    }


@router.get("/connections/{user_id}")
async def check_user_connection(user_id: str):
    """
    Check if a specific user has an active WebSocket connection.
    """
    is_connected = connection_manager.is_user_connected(user_id)
    connection_count = len(connection_manager.active_connections.get(user_id, []))
    
    return {
        "user_id": user_id,
        "is_connected": is_connected,
        "connection_count": connection_count
    }
