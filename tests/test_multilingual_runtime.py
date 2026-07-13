from pathlib import Path
from types import SimpleNamespace

from chaturbo_espanol_runtime import (
    AR_MERGED_RELATIVE_PATH,
    ChaturboEspanolRuntime,
    CONTINUAL_LORA_PROFILES,
)
from download_models import expand_profile_dependencies
from multilingual_runtime import MultilingualRuntime, normalize_spanish_text


def runtime() -> MultilingualRuntime:
    return MultilingualRuntime(SimpleNamespace())


def artifact_runtime() -> ChaturboEspanolRuntime:
    return ChaturboEspanolRuntime(SimpleNamespace())


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
    rt = artifact_runtime()
    payload = rt.profiles_payload()
    profiles = {item["id"]: item for item in payload["profiles"]}
    assert profiles["lucia-cl-pilot"]["conditioning_profile"] == "lucia-latam"
    assert profiles["lucia-co-pilot"]["conditioning_profile"] == "lucia-latam"


def test_continual_loras_declare_ar_merged_base():
    rt = artifact_runtime()
    payload = rt.profiles_payload()
    profiles = {item["id"]: item for item in payload["profiles"]}
    for profile_id in CONTINUAL_LORA_PROFILES:
        assert profiles[profile_id]["base_profile"] == "lucia-ar"
        assert profiles[profile_id]["base_checkpoint"] == AR_MERGED_RELATIVE_PATH
    assert profiles["lucia-ar"]["base_profile"] == "official-turbo"


def test_download_dependencies_include_warm_base_and_persona():
    assert expand_profile_dependencies(["lucia-latam"]) == ["lucia-ar", "lucia-latam"]
    assert expand_profile_dependencies(["lucia-cl-pilot"]) == [
        "lucia-ar",
        "lucia-latam",
        "lucia-cl-pilot",
    ]
    assert expand_profile_dependencies(["lucia-co-pilot", "lucia-ar"]) == [
        "lucia-ar",
        "lucia-latam",
        "lucia-co-pilot",
    ]
