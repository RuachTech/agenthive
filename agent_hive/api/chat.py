"""Chat-related API endpoints and functionality."""

import uuid
import json
from typing import Dict, Any, List, Optional, Union, AsyncGenerator, TYPE_CHECKING
import structlog

from fastapi import UploadFile
from fastapi.responses import StreamingResponse

from agent_hive.core.multimodal import FileValidationError, FileProcessingError
from agent_hive.api.validation import (
    RequestValidator,
    ResponseFormatter,
    APIError,
    ValidationError,
    ServiceUnavailableError,
    create_http_exception,
    sanitize_input,
)

if TYPE_CHECKING:
    from agent_hive.api.core import AgentHiveAPI

logger = structlog.get_logger()


class ChatService:
    """Service class for chat-related operations."""

    def __init__(self, api_instance: "AgentHiveAPI") -> None:
        self.api = api_instance

    async def direct_chat(
        self,
        agent_name: str,
        message: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
        files: Optional[List[UploadFile]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Direct chat endpoint with agent selection and message routing.

        Args:
            agent_name: Name of the agent to chat with
            message: User message
            session_id: Optional session ID (will create new if not provided)
            user_id: User identifier
            files: Optional uploaded files
            stream: Whether to stream the response

        Returns:
            Response from the agent or streaming response
        """
        try:
            # Validate inputs
            available_agents = self.api.agent_factory.list_agents()
            agent_name = RequestValidator.validate_agent_name(
                agent_name, available_agents
            )
            message = RequestValidator.validate_message_content(sanitize_input(message))
            user_id = RequestValidator.validate_user_id(user_id)
            session_id = RequestValidator.validate_session_id(session_id)

            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())

            # Process uploaded files if any
            processed_files, file_errors = await self._process_uploaded_files(
                files, user_id
            )

            # Add multimodal content to message if files were processed
            if processed_files:
                multimodal_context = "\n\n**Uploaded Files:**\n"
                for pf in processed_files:
                    multimodal_context += (
                        f"- {pf.original_name} ({pf.file_type.value})\n"
                    )
                message = message + multimodal_context

            if stream:
                # Return streaming response
                return StreamingResponse(
                    self._stream_direct_chat(
                        session_id, agent_name, message, user_id, processed_files
                    ),
                    media_type="text/plain",
                )
            else:
                # Return regular response
                result = await self.api.graph_factory.execute_direct_chat(
                    session_id=session_id,
                    agent_name=agent_name,
                    message=message,
                    user_id=user_id,
                )

                # Add file information to response
                if processed_files or file_errors:
                    result["file_processing"] = {
                        "processed_files": [
                            {
                                "file_id": pf.file_id,
                                "name": pf.original_name,
                                "type": pf.file_type.value,
                                "status": pf.status.value,
                            }
                            for pf in processed_files
                        ],
                        "errors": file_errors,
                        "total_files": len(files) if files else 0,
                        "successful": len(processed_files),
                        "failed": len(file_errors),
                    }

                return ResponseFormatter.success_response(
                    data=result, message="Direct chat completed successfully"
                )

        except APIError as e:
            logger.error(
                "Direct chat validation failed", agent=agent_name, error=str(e)
            )
            raise create_http_exception(e)
        except Exception as e:
            logger.error("Direct chat failed", agent=agent_name, error=str(e))
            error = ServiceUnavailableError("direct_chat", str(e))
            raise create_http_exception(error)

    async def orchestrate_task(
        self,
        task: str,
        session_id: Optional[str] = None,
        user_id: str = "default_user",
        files: Optional[List[UploadFile]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """
        Orchestrate task endpoint for complex task delegation.

        Args:
            task: Task description
            session_id: Optional session ID
            user_id: User identifier
            files: Optional uploaded files
            stream: Whether to stream the response

        Returns:
            Orchestration result or streaming response
        """
        try:
            # Validate inputs
            task = RequestValidator.validate_message_content(sanitize_input(task))
            user_id = RequestValidator.validate_user_id(user_id)
            session_id = RequestValidator.validate_session_id(session_id)

            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())

            # Process uploaded files if any
            processed_files, file_errors = await self._process_uploaded_files(
                files, user_id
            )

            # Add multimodal content to task if files were processed
            if processed_files:
                multimodal_context = "\n\n**Uploaded Files:**\n"
                for pf in processed_files:
                    multimodal_context += (
                        f"- {pf.original_name} ({pf.file_type.value})\n"
                    )
                task = task + multimodal_context

            if stream:
                return StreamingResponse(
                    self._stream_orchestration(
                        session_id, task, user_id, processed_files
                    ),
                    media_type="text/plain",
                )
            else:
                result = await self._execute_orchestration(
                    session_id, task, user_id, processed_files
                )

                # Add file processing info
                if processed_files or file_errors:
                    result["file_processing"] = {
                        "processed_files": [
                            {
                                "file_id": pf.file_id,
                                "name": pf.original_name,
                                "type": pf.file_type.value,
                                "status": pf.status.value,
                            }
                            for pf in processed_files
                        ],
                        "errors": file_errors,
                    }

                return ResponseFormatter.success_response(
                    data=result, message="Task orchestration completed successfully"
                )

        except APIError as e:
            logger.error("Task orchestration validation failed", error=str(e))
            raise create_http_exception(e)
        except Exception as e:
            logger.error("Task orchestration failed", task=task[:100], error=str(e))
            error = ServiceUnavailableError("task_orchestration", str(e))
            raise create_http_exception(error)

    async def _process_uploaded_files(
        self, files: Optional[List[UploadFile]], user_id: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process uploaded files and return results and errors."""
        processed_files = []
        file_errors = []

        if files:
            for file in files:
                try:
                    # Validate file
                    content = await file.read()
                    RequestValidator.validate_file_upload(
                        filename=file.filename or "unknown",
                        content_type=file.content_type,
                        file_size=len(content),
                    )

                    processed_file = await self.api.multimodal_processor.process_file(
                        filename=file.filename or "unknown",
                        content=content,
                        mime_type=file.content_type,
                        user_id=user_id,
                    )
                    processed_files.append(processed_file)

                except (FileValidationError, FileProcessingError, ValidationError) as e:
                    logger.warning(
                        "File processing failed", filename=file.filename, error=str(e)
                    )
                    file_errors.append(
                        {
                            "filename": file.filename,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    )
                    continue

        return processed_files, file_errors

    async def _stream_direct_chat(
        self,
        session_id: str,
        agent_name: str,
        message: str,
        user_id: str,
        processed_files: List[Any],
    ) -> AsyncGenerator[str, None]:
        """Stream direct chat responses."""
        try:
            async for chunk in self.api.graph_factory.stream_direct_chat(
                session_id=session_id,
                agent_name=agent_name,
                message=message,
                user_id=user_id,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            error_chunk = {"type": "error", "error": str(e), "session_id": session_id}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    async def _execute_orchestration(
        self, session_id: str, task: str, user_id: str, processed_files: List[Any]
    ) -> Dict[str, Any]:
        """Execute orchestration workflow."""
        try:
            # Get or create session
            session = await self.api.session_manager.get_session(session_id)
            if not session:
                session = await self.api.session_manager.create_session(
                    session_id=session_id,
                    user_id=user_id,
                    mode="orchestration",
                    initial_task=task,
                )

            # Update session with task and files
            from langchain_core.messages import HumanMessage

            session.state["messages"].append(HumanMessage(content=task))
            session.state["task"] = task

            # Add processed files to session
            if processed_files:
                session.multimodal_files.extend(processed_files)
                session.state["multimodal_content"] = {
                    pf.file_id: pf.processed_content for pf in processed_files
                }

            # Get orchestrator graph
            orchestrator_graph = (
                await self.api.orchestrator_factory.create_orchestrator_graph()
            )

            # Execute orchestration
            from langchain_core.runnables import RunnableConfig

            thread_config = RunnableConfig(configurable={"thread_id": session_id})

            result = await orchestrator_graph.ainvoke(
                session.state, config=thread_config
            )

            # Update session with result
            await self.api.session_manager.update_session_state(session_id, result)

            # Extract response
            response_content = ""
            if result.get("messages"):
                last_message = result["messages"][-1]
                response_content = getattr(last_message, "content", str(last_message))

            return {
                "response": response_content,
                "session_id": session_id,
                "mode": "orchestration",
                "status": "success",
                "participating_agents": result.get("active_agents", []),
                "task_status": result.get("task_status", {}),
            }

        except Exception as e:
            logger.error("Orchestration execution failed", error=str(e))
            raise

    async def _stream_orchestration(
        self, session_id: str, task: str, user_id: str, processed_files: List[Any]
    ) -> AsyncGenerator[str, None]:
        """Stream orchestration responses."""
        try:
            # This is a simplified streaming implementation
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            result = await self._execute_orchestration(
                session_id, task, user_id, processed_files
            )

            yield f"data: {json.dumps({'type': 'content', 'content': result['response']})}\n\n"
            yield f"data: {json.dumps({'type': 'complete', 'result': result})}\n\n"

        except Exception as e:
            error_chunk = {"type": "error", "error": str(e), "session_id": session_id}
            yield f"data: {json.dumps(error_chunk)}\n\n"
