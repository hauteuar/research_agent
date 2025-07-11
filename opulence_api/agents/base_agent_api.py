# agents/base_agent_api.py
"""
API-Compatible Base Agent for Opulence System
Provides common functionality for all agents using API calls instead of direct GPU management
"""

import weakref
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, Optional
import time

class BaseOpulenceAgent:
    """Base class for all API-based Opulence agents"""

    def __init__(self, coordinator, agent_type: str, db_path: str = "opulence_data.db", gpu_id: int = 0):
        self.coordinator = coordinator
        self.agent_type = agent_type
        self.db_path = db_path
        self.gpu_id = gpu_id  # Preferred GPU for API calls
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")

        # API-specific tracking
        self._api_calls_made = 0
        self._total_api_time = 0.0
        self._last_api_call_time = None
        self._initialization_time = time.time()

        # Agent state
        self._is_active = True

        # API parameters for this agent
        self.api_params = {
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9
        }

        # Register for automatic cleanup
        if coordinator:
            weakref.finalize(self, self._cleanup_callback, agent_type)

    @staticmethod
    def _cleanup_callback(agent_type: str):
        """Static cleanup callback for weakref"""
        try:
            logging.getLogger(__name__).info(f"🧹 Auto-cleaned {agent_type} agent")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Auto-cleanup failed for {agent_type}: {e}")

    @asynccontextmanager
    async def get_engine_context(self):
        """API-compatible context manager - returns API client interface"""
        try:
            # Create API context wrapper
            api_context = APIEngineContext(self.coordinator, self.gpu_id)
            yield api_context
        except Exception as e:
            self.logger.error(f"API context error in {self.agent_type}: {e}")
            raise
        finally:
            # No cleanup needed for API calls
            pass

    async def call_api(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """Make API call through coordinator with tracking"""
        start_time = time.time()

        try:
            # Merge agent-specific params with call-specific params
            final_params = self.api_params.copy()
            if params:
                final_params.update(params)

            # Make API call through coordinator
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=final_params,
                preferred_gpu_id=self.gpu_id
            )

            # Extract text from API response
            if isinstance(result, dict):
                response_text = result.get('text', result.get('response', ''))
            else:
                response_text = str(result)

            # Track API call statistics
            api_time = time.time() - start_time
            self._api_calls_made += 1
            self._total_api_time += api_time
            self._last_api_call_time = time.time()

            self.logger.debug(f"API call completed in {api_time:.2f}s for {self.agent_type}")

            return response_text

        except Exception as e:
            self.logger.error(f"API call failed for {self.agent_type}: {str(e)}")
            raise RuntimeError(f"API call failed: {str(e)}")

    async def get_engine(self):
        """DEPRECATED: For backwards compatibility - use get_engine_context() instead"""
        self.logger.warning(f"⚠️ {self.agent_type} using deprecated get_engine() - use call_api() instead")

        # Return a mock engine that delegates to API calls
        return MockEngine(self)

    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result.update({
                'gpu_used': self.gpu_id,
                'agent_type': self.agent_type,
                'api_based': True,
                'api_calls_made': self._api_calls_made,
                'total_api_time': self._total_api_time,
                'average_api_time': self._total_api_time / max(self._api_calls_made, 1),
                'coordinator_type': getattr(self.coordinator, 'stats', {}).get('coordinator_type', 'api_based')
            })
        return result

    def cleanup(self):
        """Manual cleanup method"""
        self.logger.info(f"🧹 Cleaning up {self.agent_type} agent...")

        # Mark as inactive
        self._is_active = False

        # Log final statistics
        if self._api_calls_made > 0:
            avg_time = self._total_api_time / self._api_calls_made
            uptime = time.time() - self._initialization_time
            self.logger.info(f"✅ {self.agent_type} final stats: {self._api_calls_made} API calls, "
                           f"avg {avg_time:.2f}s, uptime {uptime:.1f}s")

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        uptime = time.time() - self._initialization_time
        avg_api_time = self._total_api_time / max(self._api_calls_made, 1)

        return {
            "agent_type": self.agent_type,
            "gpu_id": self.gpu_id,
            "api_based": True,
            "is_active": self._is_active,
            "api_calls_made": self._api_calls_made,
            "total_api_time": self._total_api_time,
            "average_api_time": avg_api_time,
            "last_api_call_time": self._last_api_call_time,
            "uptime_seconds": uptime,
            "calls_per_minute": (self._api_calls_made / max(uptime / 60, 1)) if uptime > 0 else 0
        }

    def update_api_params(self, **kwargs):
        """Update API parameters for this agent"""
        self.api_params.update(kwargs)
        self.logger.info(f"Updated API params for {self.agent_type}: {kwargs}")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"type={self.agent_type}, "
                f"gpu={self.gpu_id}, "
                f"api_calls={self._api_calls_made}, "
                f"active={self._is_active})")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class APIEngineContext:
    """API-compatible engine context that mimics the original engine interface"""

    def __init__(self, coordinator, preferred_gpu_id: int = None):
        self.coordinator = coordinator
        self.preferred_gpu_id = preferred_gpu_id
        self.logger = logging.getLogger(f"{__name__}.APIEngineContext")

    async def generate(self, prompt: str, sampling_params, request_id: str = None):
        """Generate text via API (compatible with original engine interface)"""
        # Convert sampling_params to API parameters
        params = {
            "max_tokens": getattr(sampling_params, 'max_tokens', 512),
            "temperature": getattr(sampling_params, 'temperature', 0.7),
            "top_p": getattr(sampling_params, 'top_p', 0.9),
            "top_k": getattr(sampling_params, 'top_k', 50),
            "frequency_penalty": getattr(sampling_params, 'frequency_penalty', 0.0),
            "presence_penalty": getattr(sampling_params, 'presence_penalty', 0.0),
            "stop": getattr(sampling_params, 'stop', None),
            "seed": getattr(sampling_params, 'seed', None),
        }

        # Call API
        result = await self.coordinator.call_model_api(
            prompt=prompt,
            params=params,
            preferred_gpu_id=self.preferred_gpu_id
        )

        # Convert API response to engine-compatible format
        class MockOutput:
            def __init__(self, text: str, finish_reason: str):
                self.text = text
                self.finish_reason = finish_reason
                self.token_ids = []  # Not available from API

        class MockRequestOutput:
            def __init__(self, result: Dict[str, Any]):
                self.outputs = [MockOutput(
                    result.get('text', ''),
                    result.get('finish_reason', 'stop')
                )]
                self.finished = True
                self.prompt_token_ids = []  # Not available from API

        # Yield result (to match async generator interface)
        yield MockRequestOutput(result)


class MockEngine:
    """Mock engine for backwards compatibility with agents expecting engine interface"""

    def __init__(self, agent):
        self.agent = agent

    async def generate(self, prompt: str, sampling_params=None, request_id: str = None):
        """Mock generate method that uses API calls"""
        # Convert sampling params to API params
        params = {}
        if sampling_params:
            params = {
                "max_tokens": getattr(sampling_params, 'max_tokens', 512),
                "temperature": getattr(sampling_params, 'temperature', 0.7),
            }

        # Use agent's API call method
        response_text = await self.agent.call_api(prompt, params)

        # Mock the expected response format
        class MockOutput:
            def __init__(self, text):
                self.text = text
                self.finish_reason = "stop"

        class MockRequestOutput:
            def __init__(self, text):
                self.outputs = [MockOutput(text)]
                self.finished = True

        yield MockRequestOutput(response_text)


# ==================== Utility Functions ====================

def create_api_agent(coordinator, agent_class, agent_type: str, **kwargs):
    """Factory function to create API-based agents"""
    return agent_class(
        coordinator=coordinator,
        llm_engine=None,  # No direct engine for API agents
        **kwargs
    )

async def test_api_agent_functionality(coordinator):
    """Test function to verify API agent functionality"""

    # Create a test agent
    test_agent = BaseOpulenceAgent(coordinator, "test_agent", gpu_id=1)

    try:
        # Test API call
        response = await test_agent.call_api("What is 2+2?", {"max_tokens": 50})
        print(f"API Response: {response}")

        # Test context manager
        async with test_agent.get_engine_context() as engine_context:
            print("Engine context acquired successfully")

        # Get stats
        stats = test_agent.get_agent_stats()
        print(f"Agent stats: {stats}")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False

    finally:
        test_agent.cleanup()

# ==================== Example Usage ====================

async def example_api_agent_usage():
    """Example of using the API-compatible base agent"""

    # Assuming you have an API coordinator
    from ..api_opulence_coordinator import create_api_coordinator_from_endpoints

    gpu_endpoints = {
        1: "http://localhost:8000",
        2: "http://localhost:8001"
    }

    coordinator = create_api_coordinator_from_endpoints(gpu_endpoints)
    await coordinator.initialize()

    try:
        # Create API-based agent
        agent = BaseOpulenceAgent(coordinator, "example_agent", gpu_id=1)

        # Make API calls
        response1 = await agent.call_api("Explain COBOL programming")
        print(f"Response 1: {response1[:100]}...")

        response2 = await agent.call_api("What is data lineage?")
        print(f"Response 2: {response2[:100]}...")

        # Use context manager (for compatibility)
        async with agent.get_engine_context() as api_context:
            print("Using API context successfully")

        # Get final statistics
        stats = agent.get_agent_stats()
        print(f"Final agent stats: {stats}")

    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_api_agent_usage())