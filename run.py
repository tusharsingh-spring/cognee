"""
ARGUS V3 — Unified Launcher
Starts the video processing pipeline AND the chat interface together.
Run: python run.py --video auto
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def launch_chat(port: int = 8501):
    chat_path = ROOT / "chat_ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(chat_path),
         "--server.headless", "true",
         "--server.port", str(port),
         "--server.fileWatcherType", "none",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
    )


def launch_dashboard(port: int = 8502):
    dash_path = ROOT / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dash_path),
         "--server.headless", "true",
         "--server.port", str(port),
         "--server.fileWatcherType", "none",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ARGUS V3 — Unified Pipeline + Chat")
    parser.add_argument("--video", type=str, default=None, help="Video source (auto/list/PATH)")
    parser.add_argument("--webcam", action="store_true", help="Use local webcam (camera index 0)")
    parser.add_argument("--camera", type=str, default=None, help="Camera URL or index (e.g. 0, http://...)")
    parser.add_argument("--turbo", action="store_true", help="Skip VLM + LLM")
    parser.add_argument("--headless", action="store_true", help="Run pipeline without display")
    parser.add_argument("--chat-only", action="store_true", help="Only launch chat (no pipeline)")
    parser.add_argument("--dashboard-only", action="store_true", help="Only launch dashboard")
    parser.add_argument("--no-chat", action="store_true", help="Skip chat UI, pipeline only")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard")
    args = parser.parse_args()

    if args.chat_only:
        print("Launching Chat UI at http://localhost:8501")
        launch_chat()
        return

    if args.dashboard_only:
        print("Launching Dashboard at http://localhost:8502")
        launch_dashboard()
        return

    print("=" * 60)
    print("ARGUS V3 — Unified Launcher")
    print("=" * 60)
    print()

    threads = []

    if not args.no_chat:
        print("  Chat UI:      http://localhost:8501")
        chat_thread = threading.Thread(target=launch_chat, args=(8501,), daemon=True)
        threads.append(("Chat UI", chat_thread))

    if not args.no_dashboard:
        print("  Dashboard:    http://localhost:8502")
        dash_thread = threading.Thread(target=launch_dashboard, args=(8502,), daemon=True)
        threads.append(("Dashboard", dash_thread))

    print("  Pipeline:     processing video...")
    print()

    for name, t in threads:
        t.start()

    time.sleep(3)

    pipeline_args = [sys.executable, str(ROOT / "main.py")]
    if args.webcam:
        pipeline_args.append("--webcam")
    elif args.camera:
        pipeline_args.extend(["--video", args.camera])
    elif args.video:
        pipeline_args.extend(["--video", args.video])
    else:
        pipeline_args.extend(["--video", "auto"])
    if args.turbo:
        pipeline_args.append("--turbo")
    if args.headless:
        pipeline_args.append("--headless")

    print(f"Pipeline: {' '.join(pipeline_args)}")
    print("=" * 60)
    print()

    try:
        subprocess.run(pipeline_args, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\nShutting down...")

    if threads:
        print(f"\nChat UI still available at http://localhost:8501")
        print("Press Ctrl+C to stop everything.")
        try:
            while any(t.is_alive() for _, t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print("ARGUS stopped.")


if __name__ == "__main__":
    main()
