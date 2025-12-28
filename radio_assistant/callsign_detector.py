"""Callsign detection in transcribed text."""

from dataclasses import dataclass
import re
from loguru import logger


@dataclass
class DetectionResult:
    """Result from callsign detection."""

    detected: bool
    confidence: float
    matched_form: str | None
    dispatch_keyword_present: bool


class CallsignDetector:
    """Detect amateur radio callsigns in transcribed text."""

    # Phonetic alphabet mapping
    PHONETIC_MAP = {
        "A": ["alpha"],
        "B": ["bravo"],
        "C": ["charlie"],
        "D": ["delta"],
        "E": ["echo"],
        "F": ["foxtrot"],
        "G": ["golf"],
        "H": ["hotel"],
        "I": ["india"],
        "J": ["juliet"],
        "K": ["kilo"],
        "L": ["lima"],
        "M": ["mike"],
        "N": ["november"],
        "O": ["oscar"],
        "P": ["papa"],
        "Q": ["quebec"],
        "R": ["romeo"],
        "S": ["sierra"],
        "T": ["tango"],
        "U": ["uniform"],
        "V": ["victor"],
        "W": ["whiskey"],
        "X": ["xray", "x-ray"],
        "Y": ["yankee"],
        "Z": ["zulu"],
        "0": ["zero"],
        "1": ["one"],
        "2": ["two"],
        "3": ["three"],
        "4": ["four"],
        "5": ["five"],
        "6": ["six"],
        "7": ["seven"],
        "8": ["eight"],
        "9": ["nine"],
    }

    # Dispatch keywords
    DISPATCH_KEYWORDS = [
        "dispatch",
        "dispatched",
        "calling",
        "come in",
        "do you copy",
        "radio check",
    ]

    def __init__(
        self,
        callsign: str,
        require_dispatch_keyword: bool = True,
        phonetic_alphabet: bool = True,
    ):
        """Initialize callsign detector.

        Args:
            callsign: The callsign to detect (e.g., "WSJJ659")
            require_dispatch_keyword: Require dispatch keyword for detection
            phonetic_alphabet: Enable phonetic alphabet matching
        """
        self.callsign = callsign.upper()
        self.require_dispatch = require_dispatch_keyword
        self.phonetic = phonetic_alphabet

        # Build phonetic patterns
        self._phonetic_patterns = []
        if self.phonetic:
            self._phonetic_patterns = self._build_phonetic_patterns()

        logger.debug(
            f"Initialized CallsignDetector for '{self.callsign}' "
            f"(require_dispatch={require_dispatch_keyword}, phonetic={phonetic_alphabet})"
        )

    def _build_phonetic_patterns(self) -> list[str]:
        """Build phonetic variations of the callsign.

        Returns:
            List of phonetic pattern strings
        """
        patterns = []

        # Build phonetic pattern for each character
        for char in self.callsign:
            if char in self.PHONETIC_MAP:
                patterns.append(self.PHONETIC_MAP[char])
            else:
                patterns.append([char])

        # Generate all combinations (for simplicity, just use first variant)
        phonetic_words = [variants[0] for variants in patterns]

        # Create pattern with optional spaces/punctuation between words
        return [" ".join(phonetic_words)]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching.

        Args:
            text: Input text

        Returns:
            Normalized text (uppercase, stripped, punctuation removed)
        """
        # Convert to uppercase
        text = text.upper()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Keep alphanumeric and spaces
        text = re.sub(r"[^A-Z0-9\s]", " ", text)

        return text.strip()

    def _check_dispatch_keyword(self, text: str) -> bool:
        """Check if dispatch keyword is present.

        Args:
            text: Normalized text

        Returns:
            True if any dispatch keyword found
        """
        text_lower = text.lower()
        for keyword in self.DISPATCH_KEYWORDS:
            if keyword in text_lower:
                return True
        return False

    def _check_direct_match(self, text: str) -> str | None:
        """Check for direct callsign match (with spacing/punctuation variations).

        Args:
            text: Normalized text

        Returns:
            Matched form if found, None otherwise
        """
        # Remove all spaces from both callsign and text for flexible matching
        callsign_no_space = self.callsign.replace(" ", "")

        # Check for exact match
        if callsign_no_space in text.replace(" ", ""):
            return self.callsign

        # Check with various spacing patterns
        # e.g., "W S J J 6 5 9" or "W-S-J-J-6-5-9"
        spaced_callsign = " ".join(list(self.callsign))
        if spaced_callsign in text:
            return spaced_callsign

        return None

    def _check_phonetic_match(self, text: str) -> str | None:
        """Check for phonetic alphabet match.

        Args:
            text: Normalized text

        Returns:
            Matched phonetic form if found, None otherwise
        """
        if not self.phonetic or not self._phonetic_patterns:
            return None

        text_lower = text.lower()
        for pattern in self._phonetic_patterns:
            if pattern in text_lower:
                return pattern

        return None

    def detect(self, transcription: str) -> DetectionResult:
        """Detect callsign in transcription.

        Args:
            transcription: Transcribed text to search

        Returns:
            DetectionResult with detection status and details
        """
        # Normalize text
        normalized = self._normalize_text(transcription)

        # Check for dispatch keyword
        dispatch_present = self._check_dispatch_keyword(transcription)

        # Check for callsign matches
        direct_match = self._check_direct_match(normalized)
        phonetic_match = self._check_phonetic_match(transcription)

        # Determine if detected
        matched_form = direct_match or phonetic_match
        detected = matched_form is not None

        # Apply dispatch keyword requirement
        if self.require_dispatch and not dispatch_present:
            detected = False

        # Calculate confidence
        # Simple heuristic: higher confidence for direct match, lower for phonetic
        confidence = 0.0
        if detected:
            if direct_match:
                confidence = 0.95
            elif phonetic_match:
                confidence = 0.85

            # Reduce confidence if no dispatch keyword
            if not dispatch_present:
                confidence *= 0.5

        logger.debug(
            f"Detection: {detected} (confidence={confidence:.2f}, "
            f"form='{matched_form}', dispatch={dispatch_present})"
        )

        return DetectionResult(
            detected=detected,
            confidence=confidence,
            matched_form=matched_form,
            dispatch_keyword_present=dispatch_present,
        )
