"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import signal
import asyncio
import logging
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from alita_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from alita_app.moves import MovementManager
    from alita_app.console import LocalStream
    from alita_app.tools.core_tools import ToolDependencies
    from alita_app.anthropic_handler import AnthropicHandler
    from alita_app.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Auto-enable Gradio in simulation mode (both MuJoCo for daemon and mockup-sim for desktop app)
    status = robot.client.get_status()
    if isinstance(status, dict):
        simulation_enabled = status.get("simulation_enabled", False)
        mockup_sim_enabled = status.get("mockup_sim_enabled", False)
    else:
        simulation_enabled = getattr(status, "simulation_enabled", False)
        mockup_sim_enabled = getattr(status, "mockup_sim_enabled", False)

    is_simulation = simulation_enabled or mockup_sim_enabled

    if is_simulation and not args.gradio:
        logger.info("Simulation mode detected. Automatically enabling gradio flag.")
        args.gradio = True

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = AnthropicHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)

    stream_manager: gr.Blocks | LocalStream | None = None
    http_app: Optional[FastAPI] = None  # FastAPI to serve via uvicorn (gradio mode only)

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="Anthropic API Key",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY") if not get_space() else "",
        )

        from alita_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        _mount_wake_routes(app, handler)
        _mount_handler_startup(app, handler)
        http_app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )
        # Expose wake/sleep endpoints. Prefer the ReachyMiniApp runtime's
        # settings_app when available; otherwise spin up our own on 7860.
        if settings_app is not None:
            _mount_wake_routes(settings_app, handler)
        else:
            _start_standalone_wake_server(handler, port=7860)

    # Universal manual-wake fallback (works in any mode).
    _install_wake_signal_handlers(handler)

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        if http_app is not None:
            import uvicorn
            uvicorn.run(http_app, host="0.0.0.0", port=7860, log_level="warning")
        else:
            stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


def _mount_wake_routes(app: "FastAPI", handler: "Any") -> None:
    """Register POST /api/wake and POST /api/sleep on the FastAPI app.

    Wake  = HIBERNATE → CONVERSATION (open multi-turn session).
    Sleep = whatever state → HIBERNATE (mic muted, no movement).

    Default state at app launch is CONVERSATION, so wake is only useful
    after an explicit sleep.

    Trigger from the shell:
      curl -X POST http://localhost:7860/api/wake
      curl -X POST http://localhost:7860/api/sleep
    """
    @app.post("/api/wake")
    async def _wake() -> Dict[str, str]:
        await handler.wake()
        return {"state": handler.state_machine.state.value}

    @app.post("/api/sleep")
    async def _sleep() -> Dict[str, str]:
        await handler.force_sleep()
        return {"state": handler.state_machine.state.value}


def _mount_handler_startup(app: "FastAPI", handler: "Any") -> None:
    """Schedule handler.start_up() on uvicorn boot.

    In gradio mode, fastrtc only calls handler.start_up() when a browser
    opens a websocket. We want the ElevenLabs/VAD/store init to happen
    immediately so external wake (curl/SIGUSR1) and headless audio work
    without a browser. start_up() is idempotent (guards on _connected_event).
    """
    @app.on_event("startup")
    async def _bootstrap() -> None:
        asyncio.create_task(handler.start_up(), name="alita-handler-startup")


def _start_standalone_wake_server(handler: "Any", port: int = 7860) -> None:
    """Run a tiny FastAPI on its own thread for wake/sleep endpoints.

    Used in headless mode when no settings_app from ReachyMiniApp runtime is
    available — gives `curl -X POST http://localhost:7860/api/wake` a home.
    """
    import uvicorn

    app = FastAPI()
    _mount_wake_routes(app, handler)
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "0.0.0.0", "port": port, "log_level": "warning"},
        daemon=True,
    )
    thread.start()
    logging.getLogger(__name__).info(
        "Standalone wake server on http://0.0.0.0:%d/api/wake", port
    )


def _install_wake_signal_handlers(handler: "Any") -> None:
    """SIGUSR1 = wake (HIBERNATE → CONVERSATION), SIGUSR2 = sleep. Any run mode."""
    main_loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        main_loop = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        main_loop = None

    def _schedule(coro_fn: Any) -> None:
        try:
            loop = main_loop or asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(coro_fn(), loop)
        except Exception:
            pass

    def _on_usr1(_signum: int, _frame: Any) -> None:
        _schedule(handler.wake)

    def _on_usr2(_signum: int, _frame: Any) -> None:
        _schedule(handler.force_sleep)

    try:
        signal.signal(signal.SIGUSR1, _on_usr1)
        signal.signal(signal.SIGUSR2, _on_usr2)
    except (ValueError, AttributeError):
        # ValueError if not main thread (e.g. ReachyMiniApp launches us in a thread).
        pass


class AlitaApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = AlitaApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
