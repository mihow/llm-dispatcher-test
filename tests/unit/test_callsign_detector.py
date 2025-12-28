"""Unit tests for CallsignDetector."""

import pytest
from radio_assistant.callsign_detector import CallsignDetector, DetectionResult


class TestCallsignDetector:
    """Test suite for CallsignDetector class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        detector = CallsignDetector("WSJJ659")

        assert detector.callsign == "WSJJ659"
        assert detector.require_dispatch is True
        assert detector.phonetic is True

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        detector = CallsignDetector(
            "K6ABC", require_dispatch_keyword=False, phonetic_alphabet=False
        )

        assert detector.callsign == "K6ABC"
        assert detector.require_dispatch is False
        assert detector.phonetic is False

    def test_callsign_normalized_to_uppercase(self) -> None:
        """Test that callsign is normalized to uppercase."""
        detector = CallsignDetector("wsjj659")
        assert detector.callsign == "WSJJ659"

    # Positive test cases - with dispatch keyword

    def test_exact_match_with_dispatch(self) -> None:
        """Test exact callsign match with dispatch keyword."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 dispatch")

        assert result.detected is True
        assert result.matched_form == "WSJJ659"
        assert result.dispatch_keyword_present is True
        assert result.confidence > 0.9

    def test_lowercase_match_with_dispatch(self) -> None:
        """Test lowercase callsign match."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("wsjj659 dispatch")

        assert result.detected is True
        assert result.confidence > 0.9

    def test_spaced_callsign(self) -> None:
        """Test callsign with spaces."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("W S J J 6 5 9 dispatch")

        assert result.detected is True
        assert result.dispatch_keyword_present is True

    def test_dashed_callsign(self) -> None:
        """Test callsign with dashes (treated as spaces)."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("W-S-J-J-6-5-9 dispatch")

        assert result.detected is True

    def test_phonetic_alphabet_full(self) -> None:
        """Test phonetic alphabet spelling."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("Whiskey Sierra Juliet Juliet six five nine dispatch")

        assert result.detected is True
        assert result.dispatch_keyword_present is True
        # Phonetic match has slightly lower confidence
        assert 0.8 <= result.confidence <= 0.95

    def test_embedded_in_sentence(self) -> None:
        """Test callsign embedded in sentence."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("This is WSJJ659 dispatch calling")

        assert result.detected is True
        assert result.matched_form == "WSJJ659"

    def test_with_punctuation(self) -> None:
        """Test callsign with surrounding punctuation."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659, dispatch.")

        assert result.detected is True

    def test_extra_whitespace(self) -> None:
        """Test callsign with extra whitespace."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("  WSJJ659   dispatch  ")

        assert result.detected is True

    def test_calling_keyword(self) -> None:
        """Test with 'calling' dispatch keyword."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 calling")

        assert result.detected is True
        assert result.dispatch_keyword_present is True

    def test_come_in_keyword(self) -> None:
        """Test with 'come in' dispatch keyword."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 come in")

        assert result.detected is True
        assert result.dispatch_keyword_present is True

    def test_do_you_copy_keyword(self) -> None:
        """Test with 'do you copy' dispatch keyword."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 do you copy")

        assert result.detected is True
        assert result.dispatch_keyword_present is True

    # Negative test cases

    def test_no_dispatch_keyword(self) -> None:
        """Test callsign without dispatch keyword (should fail with default settings)."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659")

        assert result.detected is False
        assert result.dispatch_keyword_present is False

    def test_no_dispatch_keyword_allowed(self) -> None:
        """Test callsign without dispatch keyword when not required."""
        detector = CallsignDetector("WSJJ659", require_dispatch_keyword=False)
        result = detector.detect("WSJJ659")

        assert result.detected is True
        assert result.matched_form == "WSJJ659"

    def test_different_callsign(self) -> None:
        """Test with different callsign."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("KE7XYZ dispatch")

        assert result.detected is False
        assert result.matched_form is None

    def test_incomplete_callsign(self) -> None:
        """Test with incomplete callsign."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJ659 dispatch")

        assert result.detected is False

    def test_partial_callsign(self) -> None:
        """Test with only partial callsign."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("659 dispatch")

        assert result.detected is False

    def test_similar_but_wrong(self) -> None:
        """Test with similar but incorrect callsign."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ658 dispatch")

        assert result.detected is False

    def test_space_in_middle_of_callsign(self) -> None:
        """Test with space in middle of callsign."""
        detector = CallsignDetector("WSJJ659")
        # "wsjj 659" should still match due to flexible spacing
        result = detector.detect("wsjj 659 dispatch")

        assert result.detected is True

    # Different callsigns

    def test_different_callsign_k6abc(self) -> None:
        """Test detection with K6ABC callsign."""
        detector = CallsignDetector("K6ABC")
        result = detector.detect("K6ABC dispatch")

        assert result.detected is True
        assert result.matched_form == "K6ABC"

    def test_different_callsign_ke7xyz(self) -> None:
        """Test detection with KE7XYZ callsign."""
        detector = CallsignDetector("KE7XYZ")
        result = detector.detect("This is KE7XYZ calling")

        assert result.detected is True

    # Confidence scoring

    def test_confidence_direct_match(self) -> None:
        """Test confidence score for direct match."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 dispatch")

        assert result.confidence >= 0.9

    def test_confidence_phonetic_match(self) -> None:
        """Test confidence score for phonetic match."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("Whiskey Sierra Juliet Juliet six five nine dispatch")

        # Phonetic should have slightly lower confidence
        assert 0.8 <= result.confidence < 0.95

    def test_confidence_without_dispatch(self) -> None:
        """Test confidence when dispatch keyword not required."""
        detector = CallsignDetector("WSJJ659", require_dispatch_keyword=False)
        result = detector.detect("WSJJ659")

        # Should have reduced confidence without dispatch keyword
        assert result.confidence < 0.9

    # Phonetic alphabet disabled

    def test_phonetic_disabled_no_match(self) -> None:
        """Test that phonetic doesn't match when disabled."""
        detector = CallsignDetector("WSJJ659", phonetic_alphabet=False)
        result = detector.detect("Whiskey Sierra Juliet Juliet six five nine dispatch")

        assert result.detected is False

    def test_phonetic_disabled_direct_still_works(self) -> None:
        """Test that direct match still works with phonetic disabled."""
        detector = CallsignDetector("WSJJ659", phonetic_alphabet=False)
        result = detector.detect("WSJJ659 dispatch")

        assert result.detected is True

    # Edge cases

    def test_empty_string(self) -> None:
        """Test with empty string."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("")

        assert result.detected is False
        assert result.matched_form is None

    def test_only_dispatch_keyword(self) -> None:
        """Test with only dispatch keyword, no callsign."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("dispatch")

        assert result.detected is False

    def test_multiple_callsigns(self) -> None:
        """Test with multiple callsigns in text."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 calling K6ABC dispatch")

        assert result.detected is True
        assert result.matched_form == "WSJJ659"

    def test_case_insensitive_dispatch_keyword(self) -> None:
        """Test that dispatch keyword matching is case-insensitive."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect("WSJJ659 DISPATCH")

        assert result.detected is True
        assert result.dispatch_keyword_present is True

    @pytest.mark.parametrize(
        "text,expected_detected,description",
        [
            ("WSJJ659 dispatch", True, "exact match"),
            ("wsjj659 dispatch", True, "lowercase"),
            ("W S J J 6 5 9 dispatch", True, "spaced"),
            ("W-S-J-J-6-5-9 dispatch", True, "dashed"),
            (
                "Whiskey Sierra Juliet Juliet six five nine dispatch",
                True,
                "phonetic",
            ),
            ("This is WSJJ659 dispatch calling", True, "embedded"),
            ("WSJJ659 dispatch, radio check", True, "with punctuation"),
            ("WSJJ659", False, "no dispatch keyword"),
            ("KE7XYZ dispatch", False, "different callsign"),
            ("WSJ659 dispatch", False, "incomplete callsign"),
            ("659 dispatch", False, "partial callsign"),
            ("WSJJ658 dispatch", False, "similar but wrong"),
            ("  WSJJ659   dispatch  ", True, "extra whitespace"),
            ("WSJJ659, dispatch.", True, "punctuation"),
            ("wsjj 659 dispatch", True, "space in callsign"),
        ],
    )
    def test_comprehensive_cases(
        self, text: str, expected_detected: bool, description: str
    ) -> None:
        """Comprehensive test cases from implementation plan."""
        detector = CallsignDetector("WSJJ659")
        result = detector.detect(text)

        assert result.detected == expected_detected, f"Failed on: {description} ('{text}')"


class TestDetectionResult:
    """Test suite for DetectionResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a DetectionResult."""
        result = DetectionResult(
            detected=True,
            confidence=0.95,
            matched_form="WSJJ659",
            dispatch_keyword_present=True,
        )

        assert result.detected is True
        assert result.confidence == 0.95
        assert result.matched_form == "WSJJ659"
        assert result.dispatch_keyword_present is True

    def test_negative_result(self) -> None:
        """Test creating a negative detection result."""
        result = DetectionResult(
            detected=False,
            confidence=0.0,
            matched_form=None,
            dispatch_keyword_present=False,
        )

        assert result.detected is False
        assert result.matched_form is None
