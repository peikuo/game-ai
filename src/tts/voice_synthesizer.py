"""
Text-to-speech functionality for agent vocalization with different voices.
"""

import logging
import os

try:
    import pyttsx3

    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning(
        "pyttsx3 not installed, some TTS functionality will be limited")
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceSynthesizer:
    """
    Class for synthesizing speech for different agents with unique voices.
    Supports multiple TTS engines including edge-tts, pyttsx3,
    and cloud TTS services.
    """

    def __init__(self, config=None):
        """
        Initialize the VoiceSynthesizer.

        Args:
            config (dict): Configuration for the synthesizer
        """
        self.config = config or {}
        self.engine = self.config.get("engine", "edge-tts")
        self.output_dir = Path(self.config.get("output_dir", "tts_output"))
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Voice profiles for different agents
        self.voice_profiles = self.config.get(
            "voice_profiles",
            {
                "StrategicAdvisor": {
                    "voice": "en-US-ChristopherNeural",
                    "rate": "+0%",
                    "pitch": "+0Hz",
                    "style": "serious",
                },
                "MilitaryCommander": {
                    "voice": "en-US-GuyNeural",
                    "rate": "+10%",
                    "pitch": "-2Hz",
                    "style": "commanding",
                },
                "EconomicAdvisor": {
                    "voice": "en-US-AriaNeural",
                    "rate": "+0%",
                    "pitch": "+0Hz",
                    "style": "analytical",
                },
                "DiplomaticEnvoy": {
                    "voice": "en-US-JennyNeural",
                    "rate": "-5%",
                    "pitch": "+0Hz",
                    "style": "friendly",
                },
                "ChiefExecutive": {
                    "voice": "en-US-DavisNeural",
                    "rate": "+0%",
                    "pitch": "+0Hz",
                    "style": "confident",
                },
                "default": {
                    "voice": "en-US-ChristopherNeural",
                    "rate": "+0%",
                    "pitch": "+0Hz",
                    "style": "neutral",
                },
            },
        )

        # Queue for asynchronous TTS processing
        self.tts_queue = queue.Queue()
        self.is_processing = False
        self.processing_thread = None

        # Check if edge-tts is installed
        if self.engine == "edge-tts":
            try:
                subprocess.run(
                    ["edge-tts", "--version"], capture_output=True, check=True
                )
                logger.info("edge-tts is available")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("edge-tts not found, falling back to pyttsx3")
                self.engine = "pyttsx3"

        # Initialize pyttsx3 if needed
        if self.engine == "pyttsx3":
            if PYTTSX3_AVAILABLE:
                self.pyttsx3_engine = pyttsx3.init()
                logger.info("pyttsx3 initialized")
            else:
                logger.warning(
                    "pyttsx3 not installed, TTS functionality will be limited"
                )

    def synthesize(
            self,
            text,
            agent_name="default",
            output_file=None,
            blocking=False):
        """
        Synthesize speech for the given text using the agent's voice profile.

        Args:
            text (str): Text to synthesize
            agent_name (str): Name of the agent to use voice profile for
            output_file (str, optional): Path to save the audio file
            blocking (bool): Whether to wait for synthesis to complete

        Returns:
            str: Path to the generated audio file
        """
        if not text:
            logger.warning("Empty text provided for synthesis")
            return None

        # Get voice profile for the agent
        profile = self.voice_profiles.get(
            agent_name, self.voice_profiles["default"])

        # Generate output filename if not provided
        if not output_file:
            timestamp = int(time.time())
            output_file = self.output_dir / f"{agent_name}_{timestamp}.mp3"

        # Add to queue for processing
        task = {
            "text": text,
            "profile": profile,
            "output_file": str(output_file),
            "agent_name": agent_name,
        }

        if blocking:
            return self._process_tts_task(task)
        else:
            self.tts_queue.put(task)
            self._ensure_processing_thread()
            return str(output_file)

    def _ensure_processing_thread(self):
        """
        Ensure the TTS processing thread is running.
        """
        if not self.is_processing or (
            self.processing_thread and not self.processing_thread.is_alive()
        ):
            self.is_processing = True
            self.processing_thread = threading.Thread(
                target=self._process_tts_queue)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def _process_tts_queue(self):
        """
        Process the TTS queue in a background thread.
        """
        while self.is_processing:
            try:
                if self.tts_queue.empty():
                    time.sleep(0.1)
                    continue

                task = self.tts_queue.get(timeout=0.5)
                self._process_tts_task(task)
                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error processing TTS task: {e}")

    def _process_tts_task(self, task):
        """
        Process a single TTS task.

        Args:
            task (dict): TTS task information

        Returns:
            str: Path to the generated audio file
        """
        text = task["text"]
        profile = task["profile"]
        output_file = task["output_file"]
        agent_name = task["agent_name"]

        try:
            if self.engine == "edge-tts":
                return self._synthesize_with_edge_tts(
                    text, profile, output_file)
            elif self.engine == "pyttsx3":
                return self._synthesize_with_pyttsx3(
                    text, profile, output_file)
            else:
                logger.warning(f"Unknown TTS engine: {self.engine}")
                return None
        except Exception as e:
            logger.exception(f"Error synthesizing speech for {agent_name}:{e}")
            return None

    def _synthesize_with_edge_tts(self, text, profile, output_file):
        """
        Synthesize speech using edge-tts.

        Args:
            text (str): Text to synthesize
            profile (dict): Voice profile
            output_file (str): Output file path

        Returns:
            str: Path to the generated audio file
        """
        voice = profile.get("voice", "en-US-ChristopherNeural")
        rate = profile.get("rate", "+0%")
        pitch = profile.get("pitch", "+0Hz")

        # Create SSML with voice characteristics
        ssml = f"""
        <speak version =
            "1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang=
    "en-US">
            <voice name="{voice}">
                <prosody rate="{rate}" pitch="{pitch}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        # Save SSML to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as temp:
            temp.write(ssml)
            ssml_path = temp.name

        try:
            # Run edge-tts
            cmd = [
                "edge-tts",
                "--file",
                ssml_path,
                "--write-media",
                output_file]

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)
            logger.debug(f"edge-tts output: {result.stdout}")

            return output_file
        finally:
            # Clean up temporary file
            if os.path.exists(ssml_path):
                os.unlink(ssml_path)

    def _synthesize_with_pyttsx3(self, text, profile, output_file):
        """
        Synthesize speech using pyttsx3.

        Args:
            text (str): Text to synthesize
            profile (dict): Voice profile
            output_file (str): Output file path

        Returns:
            str: Path to the generated audio file
        """
        if not PYTTSX3_AVAILABLE:
            logger.error(
                "Cannot synthesize with pyttsx3 as it is not installed")
            return None

        try:
            # Use default engine if not initialized
            engine = getattr(self, "pyttsx3_engine", None) or pyttsx3.init()

            # Set properties based on profile
            # Note: pyttsx3 has limited voice customization compared to
            # edge-tts
            rate_offset = 0
            if "+" in profile.get("rate", "+0%"):
                rate_offset = int(
                    profile["rate"].replace(
                        "+",
                        "").replace(
                        "%",
                        ""))
            elif "-" in profile.get("rate", "+0%"):
                rate_offset = - \
                    int(profile["rate"].replace("-", "").replace("%", ""))

            # Adjust rate (default is 200)
            engine.setProperty("rate", 200 + (rate_offset * 2))

            # Try to set voice based on gender preference
            voices = engine.getProperty("voices")
            for voice in voices:
                if (
                    "female" in profile.get("voice", "").lower()
                    and "female" in voice.name.lower()
                ):
                    engine.setProperty("voice", voice.id)
                    break
                elif (
                    "male" in profile.get("voice", "").lower()
                    and "male" in voice.name.lower()
                ):
                    engine.setProperty("voice", voice.id)
                    break

            # Save to file
            engine.save_to_file(text, output_file)
            engine.runAndWait()

            return output_file
        except Exception as e:
            logger.exception(f"Error with pyttsx3: {e}")
            return None

    def stop(self):
        """
        Stop the TTS processing thread.
        """
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # Clear the queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break
