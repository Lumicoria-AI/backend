from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import Agent, AgentCapability, AgentStatus
# Assuming these repositories and services exist
from .component_repository import component_repository
from backend.services.ai_model_service import ai_model_service
from backend.services.integration_service import integration_service

import structlog
import json

logger = structlog.get_logger()

# --- Helper Functions for Agent Workflow Execution ---

def apply_mapping(data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Applies a mapping from source keys in 'data' to target keys in the output dictionary.
    Mapping format: {"target_key": "source_key"}
    """
    output = {}
    for target_key, source_key in mapping.items():
        if source_key in data:
            output[target_key] = data[source_key]
        else:
            # Handle missing source keys - raise error, log warning, or set default
            logger.warning("Source key not found in data for mapping", source_key=source_key, data=data)
            # Optionally: output[target_key] = None # Or some default value
    return output

def update_intermediate_results(intermediate_results: Dict[str, Any], step_output: Dict[str, Any], mapping: Dict[str, str]):
    """
    Updates intermediate_results by mapping step_output keys to result keys.
    Mapping format: {"result_key": "output_key"}
    """
    for result_key, output_key in mapping.items():
        if output_key in step_output:
            intermediate_results[result_key] = step_output[output_key]
        else:
             # Handle missing output keys - log warning or skip
            logger.warning("Output key not found in step output for mapping", output_key=output_key, step_output=step_output)

def evaluate_condition(data: Dict[str, Any], condition_config: Dict[str, Any]) -> bool:
    """
    Evaluates a condition based on data and condition configuration.
    Condition config could support various types (e.g., value comparison, key existence, regex match).
    Example config: {"type": "value_equals", "key": "status", "value": "completed"}
    """
    condition_type = condition_config.get("type")
    key = condition_config.get("key")

    if condition_type == "value_equals":
        expected_value = condition_config.get("value")
        return data.get(key) == expected_value
    elif condition_type == "key_exists":
        return key in data
    # TODO: Add more condition types (e.g., greater_than, less_than, regex_match)
    else:
        logger.warning("Unknown condition type", condition_type=condition_type)
        return False # Or raise an error

class AgentUniverseRepository(BaseRepository[Agent]):
    def __init__(self):
        super().__init__("agents", Agent)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("status")
        await collection.create_index("capabilities.name")
        await collection.create_index("last_active")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("status", ASCENDING),
            ("last_active", DESCENDING)
        ])
        await collection.create_index([
            ("created_by", ASCENDING),
            ("capabilities.name", ASCENDING),
            ("status", ASCENDING)
        ])
        # Text search index for name and description
        await collection.create_index([
            ("name", "text"),
            ("description", "text")
        ])

    async def create_agent(
        self,
        name: str,
        description: Optional[str],
        agent_type,
        capabilities: List = None,
        organization_id: Optional[str] = None,
        created_by: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_public: bool = False,
        tags: Optional[List[str]] = None,
        agent_model_config: Optional[Dict[str, Any]] = None,
        initial_status: str = "active",
    ) -> Agent:
        """Create a new agent in MongoDB."""
        agent_data = {
            "name": name,
            "description": description or "",
            "agent_type": agent_type.value if hasattr(agent_type, "value") else str(agent_type),
            "capabilities": [c.value if hasattr(c, "value") else str(c) for c in (capabilities or [])],
            "configuration": configuration or {},
            "organization_id": ObjectId(organization_id) if organization_id else ObjectId(),
            "created_by": ObjectId(str(created_by)) if created_by else ObjectId(),
            "is_public": is_public,
            "tags": tags or [],
            "metadata": metadata or {},
            "state": {
                "status": initial_status,
                "error_count": 0,
                "execution_count": 0,
                "memory_usage": 0,
            },
            "usage_statistics": {
                "total_runs": 0,
                "success_rate": 0.0,
                "error_rate": 0.0,
                "average_response_time": 0.0,
            },
            "version": "1.0.0",
        }
        if agent_model_config:
            agent_data["agent_model_config"] = agent_model_config
        return await self.create(agent_data)

    async def get_agent_by_id(
        self,
        agent_id: str,
        organization_id: Optional[str] = None,
    ) -> Optional[Agent]:
        """Get an agent by ID, optionally scoped to an organization."""
        agent = await self.get_by_id(agent_id)
        if agent and organization_id:
            agent_org = getattr(agent, "organization_id", None)
            if agent_org and str(agent_org) != str(organization_id):
                return None
        return agent

    async def update_agent(
        self,
        agent_id: str,
        organization_id: Optional[str] = None,
        update_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Update an agent by ID."""
        agent = await self.get_agent_by_id(agent_id, organization_id)
        if not agent:
            return None
        data = update_data or {}
        # If status is being updated, also update state.status
        if "status" in data:
            state_update = data.pop("state", {})
            state_update["status"] = data.pop("status")
            data["state"] = {**agent.state.model_dump(), **state_update}
        return await self.update(agent_id, data)

    async def delete_agent(
        self,
        agent_id: str,
        organization_id: Optional[str] = None,
    ) -> bool:
        """Delete an agent by ID."""
        agent = await self.get_agent_by_id(agent_id, organization_id)
        if not agent:
            return False
        return await self.delete(agent_id)

    async def discover_agents(
        self,
        organization_id: str = None,
        capabilities: Optional[List[str]] = None,
        agent_type: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        include_public: bool = True,
        min_success_rate: Optional[int] = None,
        max_error_rate: Optional[int] = None,
        tags: Optional[List[str]] = None,
        search_query: Optional[str] = None,
        created_by: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Agent]:
        """Discover agents based on capabilities, type, tags, and filters."""
        or_conditions = []
        if organization_id:
            try:
                or_conditions.append({"organization_id": ObjectId(organization_id)})
            except Exception:
                or_conditions.append({"organization_id": organization_id})
        if created_by:
            try:
                or_conditions.append({"created_by": ObjectId(created_by)})
            except Exception:
                or_conditions.append({"created_by": created_by})
        if include_public:
            or_conditions.append({"is_public": True})
        # If no conditions, match all
        filters = {"$or": or_conditions} if or_conditions else {}

        if capabilities:
            filters["capabilities"] = {"$in": capabilities}
        if agent_type:
            filters["agent_type"] = agent_type
        if status:
            filters["state.status"] = status.value if hasattr(status, "value") else status
        if min_success_rate is not None:
            filters["success_rate"] = {"$gte": min_success_rate}
        if max_error_rate is not None:
            filters["error_rate"] = {"$lte": max_error_rate}
        if tags:
            filters["tags"] = {"$in": tags}
        if search_query:
            filters["$text"] = {"$search": search_query}

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("success_rate", DESCENDING), ("usage_count", DESCENDING)]
        )

    async def get_agent_capabilities(
        self,
        organization_id: str,
        include_public: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all available agent capabilities with usage statistics."""
        filters = {
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_public": True} if include_public else {}
            ]
        }

        pipeline = [
            {"$match": filters},
            {"$unwind": "$capabilities"},
            {"$group": {
                "_id": "$capabilities.name",
                "description": {"$first": "$capabilities.description"},
                "count": {"$sum": 1},
                "agents": {
                    "$push": {
                        "id": "$_id",
                        "name": "$name",
                        "status": "$status",
                        "success_rate": "$success_rate"
                    }
                }
            }},
            {"$sort": {"count": -1}}
        ]

        return await self.aggregate(pipeline)

    async def get_agent_performance_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance statistics for agents in an organization."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["last_active"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": None,
                "total_agents": {"$sum": 1},
                "active_agents": {
                    "$sum": {"$cond": [{"$eq": ["$status", AgentStatus.ACTIVE]}, 1, 0]}
                },
                "avg_success_rate": {"$avg": "$success_rate"},
                "avg_error_rate": {"$avg": "$error_rate"},
                "avg_response_time": {"$avg": "$average_response_time"},
                "total_usage": {"$sum": "$usage_count"},
                "capability_stats": {
                    "$push": {
                        "capabilities": "$capabilities",
                        "success_rate": "$success_rate",
                        "usage_count": "$usage_count"
                    }
                }
            }},
            {"$project": {
                "_id": 0,
                "total_agents": 1,
                "active_agents": 1,
                "avg_success_rate": 1,
                "avg_error_rate": 1,
                "avg_response_time": 1,
                "total_usage": 1,
                "capability_stats": 1
            }}
        ]

        results = await self.aggregate(pipeline)
        if not results:
            return {}

        stats = results[0]
        # Process capability statistics
        capability_stats = {}
        for agent in stats["capability_stats"]:
            for capability in agent["capabilities"]:
                name = capability["name"]
                if name not in capability_stats:
                    capability_stats[name] = {
                        "count": 0,
                        "total_success_rate": 0,
                        "total_usage": 0
                    }
                capability_stats[name]["count"] += 1
                capability_stats[name]["total_success_rate"] += agent["success_rate"]
                capability_stats[name]["total_usage"] += agent["usage_count"]

        # Calculate averages for capabilities
        for name, stats in capability_stats.items():
            stats["avg_success_rate"] = stats["total_success_rate"] / stats["count"] if stats["count"] > 0 else 0
            del stats["total_success_rate"]

        stats["capability_stats"] = capability_stats
        del stats["_id"] # Ensure _id is not returned if present from aggregation
        return stats

    async def get_agent_recommendations(
        self,
        organization_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get personalized agent recommendations based on user context and history."""
        # Get user's agent usage history
        pipeline = [
            {"$match": {
                "organization_id": ObjectId(organization_id),
                "created_by": ObjectId(user_id)
            }},
            {"$lookup": {
                "from": "conversations",
                "localField": "_id",
                "foreignField": "agent_id",
                "as": "conversations"
            }},
            {"$addFields": {
                "conversation_count": {"$size": "$conversations"},
                "last_conversation": {"$max": "$conversations.created_at"}
            }},
            {"$sort": {
                "conversation_count": -1,
                "last_conversation": -1,
                "success_rate": -1
            }},
            {"$limit": limit}
        ]

        recommendations = await self.aggregate(pipeline)
        
        # Add context-based recommendations if context is provided
        if context and len(recommendations) < limit:
            context_capabilities = context.get("required_capabilities", [])
            if context_capabilities:
                additional_agents = await self.discover_agents(
                    organization_id=organization_id,
                    capabilities=context_capabilities,
                    status=AgentStatus.ACTIVE,
                    include_public=True,
                    limit=limit - len(recommendations)
                )
                recommendations.extend(additional_agents)

        return recommendations

    async def update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        response_time: int,
        error_message: Optional[str] = None
    ) -> Optional[Agent]:
        """Update agent performance metrics after an interaction."""
        agent = await self.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            return None

        # Calculate new metrics
        current_success_rate = agent.get("success_rate", 0)
        current_error_rate = agent.get("error_rate", 0)
        current_response_time = agent.get("average_response_time", 0)
        usage_count = agent.get("usage_count", 0) + 1

        # Simple average update for now. Could be more sophisticated (e.g., exponential moving average)
        new_success_rate = ((current_success_rate * (usage_count - 1)) + (100 if success else 0)) / usage_count if usage_count > 0 else 0
        new_error_rate = ((current_error_rate * (usage_count - 1)) + (100 if error_message else 0)) / usage_count if usage_count > 0 else 0
        new_response_time = ((current_response_time * (usage_count - 1)) + response_time) / usage_count if usage_count > 0 else 0

        # Update agent
        update_data = {
            "success_rate": new_success_rate,
            "error_rate": new_error_rate,
            "average_response_time": new_response_time,
            "usage_count": usage_count,
            "last_active": datetime.utcnow()
        }

        if error_message:
            update_data["$push"] = {
                "error_log": {
                    "message": error_message,
                    "timestamp": datetime.utcnow()
                }
            }

        return await self.update(agent_id, update_data)

    async def get_agent_analytics(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get detailed analytics about agent usage and performance."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["last_active"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$facet": {
                "usage_by_time": [
                    {"$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": "$last_active"
                            }
                        },
                        "count": {"$sum": 1},
                        "total_usage": {"$sum": "$usage_count"}
                    }},
                    {"$sort": {"_id": 1}}
                ],
                "capability_usage": [
                    {"$unwind": "$capabilities"},
                    {"$group": {
                        "_id": "$capabilities.name",
                        "agent_count": {"$sum": 1},
                        "total_usage": {"$sum": "$usage_count"},
                        "avg_success_rate": {"$avg": "$success_rate"}
                    }},
                    {"$sort": {"total_usage": -1}}
                ],
                "performance_metrics": [
                    {"$group": {
                        "_id": None,
                        "avg_success_rate": {"$avg": "$success_rate"},
                        "avg_error_rate": {"$avg": "$error_rate"},
                        "avg_response_time": {"$avg": "$average_response_time"},
                        "total_agents": {"$sum": 1},
                        "active_agents": {
                            "$sum": {"$cond": [{"$eq": ["$status", AgentStatus.ACTIVE]}, 1, 0]}
                        }
                    }}
                ]
            }}
        ]

        results = await self.aggregate(pipeline)
        if not results:
            return {}

        analytics = results[0]
        if analytics["performance_metrics"]:
            analytics["performance_metrics"] = analytics["performance_metrics"][0]
            del analytics["performance_metrics"]["_id"]
        return analytics

    async def execute_agent(
        self,
        agent_id: str,
        organization_id: str,
        input_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Execute a specific agent with provided input data.
        This method orchestrates the agent's workflow, potentially involving AI model calls and integrations.
        """
        agent = await self.get_by_id(agent_id)
        if not agent or str(agent.organization_id) != organization_id:
            await logger.error("Agent not found or organization mismatch", agent_id=agent_id, organization_id=organization_id)
            raise ValueError("Agent not found or access denied")

        if agent.status != AgentStatus.ACTIVE:
            await logger.warning("Attempted to execute inactive agent", agent_id=agent_id, status=agent.status)
            raise ValueError(f"Agent is not active: {agent.status}")

        await logger.info("Executing agent", agent_id=agent_id, user_id=user_id)
        start_time = datetime.utcnow()
        success = False
        error_message = None
        output_data = {}
        intermediate_results = input_data.copy() # Use input data as initial intermediate result

        try:
            # 1. Load agent workflow and components
            # Assumes agent.configuration['workflow'] is a list of component references
            workflow_steps = agent.configuration.get("workflow", [])
            loaded_components = {}

            # Fetch all components needed for the workflow in one go (optimization)
            component_ids = [step.get("component_id") for step in workflow_steps if step.get("component_id")]
            if component_ids:
                # TODO: Implement get_many or a similar method in ComponentRepository
                # components_list = await component_repository.get_many(component_ids)
                # loaded_components = {str(c.id): c for c in components_list}
                # Placeholder for now
                 await logger.warning("ComponentRepository.get_many not implemented, using dummy components.")
                 loaded_components = {cid: {"type": "dummy", "configuration": {}} for cid in component_ids}

            # 2. Execute workflow steps sequentially
            for step in workflow_steps:
                step_type = step.get("type")
                component_id = step.get("component_id")
                
                # Get component configuration, falling back to empty dict if component not found/loaded
                component_config = loaded_components.get(component_id, {}).get("configuration", {})
                
                step_config = step.get("configuration", {}) # Step-specific configuration overrides
                current_config = {**component_config, **step_config} # Merge configurations, step config overrides component config

                await logger.info("Executing workflow step", agent_id=agent_id, step_type=step_type, component_id=component_id)

                if step_type == "ai_call":
                    model_name = current_config.get("model")
                    if not model_name:
                        raise ValueError(f"AI call step missing model configuration: {step}")

                    # Apply input mapping
                    input_mapping = current_config.get("input_mapping", {})
                    ai_input_data = apply_mapping(intermediate_results, input_mapping) if input_mapping else intermediate_results
                    
                    # Assuming AI models that process text take a single string prompt
                    # If the AI model service needs a different input format, adjust this.
                    ai_input_prompt = ai_input_data.get("text", json.dumps(ai_input_data)) # Default to dumping JSON if no 'text' key

                    ai_result = await ai_model_service.process_text(model_name, ai_input_prompt, current_config.get("model_settings"))

                    # Apply output mapping
                    output_mapping = current_config.get("output_mapping", {})
                    update_intermediate_results(intermediate_results, ai_result, output_mapping)

                elif step_type == "integration_action":
                    integration_id = current_config.get("integration_id")
                    action_name = current_config.get("action")
                    if not integration_id or not action_name:
                         raise ValueError(f"Integration action step missing config: {step}")

                    # Apply input mapping for integration action
                    input_mapping = current_config.get("input_mapping", {})
                    integration_action_data = apply_mapping(intermediate_results, input_mapping) if input_mapping else intermediate_results

                    integration_result = await integration_service.execute_integration_action(
                        integration_id,
                        organization_id,
                        action_name,
                        integration_action_data # Pass mapped data
                    )

                    # Apply output mapping for integration action
                    output_mapping = current_config.get("output_mapping", {})
                    update_intermediate_results(intermediate_results, integration_result, output_mapping)

                elif step_type == "data_transformation":
                    # TODO: Implement data transformation logic based on current_config
                    await logger.info("Executing data transformation step (placeholder)", config=current_config)
                    # Example: apply simple transformation function
                    # transformation_function = get_transformation_function(current_config.get("transformation_type"))
                    # intermediate_results = transformation_function(intermediate_results, current_config.get("parameters"))
                    pass # Placeholder - direct modification of intermediate_results

                elif step_type == "conditional_logic":
                    # Evaluate the condition
                    condition_config = current_config.get("condition")
                    if not condition_config:
                        raise ValueError(f"Conditional logic step missing condition configuration: {step}")

                    condition_met = evaluate_condition(intermediate_results, condition_config)

                    # Determine the next step based on the condition result
                    if condition_met:
                        next_step_id = current_config.get("on_true")
                    else:
                        next_step_id = current_config.get("on_false")

                    if next_step_id:
                        # Find the index of the next step in the workflow
                        next_step_index = -1
                        for i, s in enumerate(workflow_steps):
                            if s.get("id") == next_step_id:
                                next_step_index = i
                                break

                        if next_step_index != -1:
                            # Jump to the next step by adjusting the loop index
                            # Need to be careful with loop control here. A while loop might be cleaner
                            # for complex branching, but for simple forward jumps, adjusting the index works.
                            # Note: This simple index adjustment does NOT support going backward or complex graph workflows.
                            # A more robust solution would use a state machine or explicit pointers.
                            # For now, let's assume simple forward linear workflows with skips.
                            await logger.info("Conditional logic jumping to step", next_step_id=next_step_id)
                            # HACK: Adjusting loop index - might need a different workflow execution model
                            # This requires the workflow steps to have IDs and be in a predictable list.
                            # It also doesn't prevent infinite loops if conditions lead back.
                            # A more robust approach is needed for production.
                            # For now, we'll break and rely on the configuration to define clear paths.
                            # TODO: Implement a proper workflow execution engine.
                            break # Break the current loop, assuming the workflow ends or continues from the API caller
                        else:
                            await logger.warning("Next step ID not found in workflow", next_step_id=next_step_id, agent_id=agent_id)
                            # Decide whether to fail or continue
                            # raise ValueError(f"Next step ID {next_step_id} not found in workflow")
                    else:
                        # No next step defined for this path, continue to the next step in sequence or end
                        pass

                # TODO: Add more step types as needed (e.g., human in the loop, notification, etc.)
                else:
                    await logger.warning("Unknown workflow step type", agent_id=agent_id, step_type=step_type)
                    # Decide whether to fail or skip unknown steps
                    # raise ValueError(f"Unknown workflow step type: {step_type}")

                # TODO: Store intermediate results or log step execution if necessary for debugging/auditing
                # await self.log_step_execution(agent_id, step.get("id"), intermediate_results, success=True)

            # 3. Final output based on intermediate_results and agent's output_schema
            # Apply final output mapping
            output_mapping = agent.output_schema.get("mapping", {})
            output_data = apply_mapping(intermediate_results, output_mapping) if output_mapping else intermediate_results

            success = True # If execution completes without exception

        except Exception as e:
            error_message = str(e)
            await logger.error("Error during agent execution", agent_id=agent_id, user_id=user_id, error=error_message)
            # TODO: Log the specific step where the error occurred
            # await self.log_step_execution(agent_id, current_step_id, intermediate_results, success=False, error=error_message)
            # Re-raise the exception after logging
            raise e

        finally:
            end_time = datetime.utcnow()
            response_time = int((end_time - start_time).total_seconds() * 1000) # in milliseconds

            # Update agent performance metrics
            await self.update_agent_performance(
                agent_id=agent_id,
                success=success,
                response_time=response_time,
                error_message=error_message
            )

        return output_data

# Create a singleton instance
agent_universe_repository = AgentUniverseRepository() 