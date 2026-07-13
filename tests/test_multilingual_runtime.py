from pathlib import Path
from types import SimpleNamespace

from multilingual_runtime import MultilingualRuntime, normalize_spanish_text


def runtime() -> MultilingualRuntime:
    return MultilingualRuntime(SimpleNamespace())


def test_spanish_normalization_preserves_language_features():
    assert normalize_spanish_text("  ¡ Hola , señor pingüino!  ") == "¡Hola, señor pingüino!"
    assert normalize_spanish_text("Mañana tomaré café") == "Mañana tomaré café."


def test_profile_aliases_are_canonicalized(monkeypatch):
    monkeypatch.setenv("DEFAULT_SPANISH_PROFILE", "lucia-ar")
    rt = runtime()
    assert rt.resolve_profile_id("Lucía") == "lucia-ar"
    assert rt.resolve_profile_id("es_419") == "lucia-latam"
    assert rt.resolve_profile_id("ES-CL") == "lucia-cl-pilot"
    assert rt.resolve_profile_id("colombia") == "lucia-co-pilot"
    assert rt.resolve_profile_id("alloy") is None


def test_internal_profile_marker_round_trip():
    rt = runtime()
    marker = rt.marker_for_profile("lucia-latam")
    assert marker == Path("/__chatterbox_profile__/lucia-latam")
    assert rt.profile_from_marker(marker) == "lucia-latam"
    assert rt.profile_from_marker(Path("/tmp/lucia-latam.wav")) is None


def test_pilot_profiles_use_latam_conditioning():
    rt = runtime()
    payload = rt.profiles_payload()
    profiles = {item["id"]: item for item in payload["profiles"]}
    assert profiles["lucia-cl-pilot"]["conditioning_profile"] == "lucia-latam"
    assert profiles["lucia-co-pilot"]["conditioning_profile"] == "lucia-latam"
