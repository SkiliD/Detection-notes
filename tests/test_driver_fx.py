"""The driver-effects note: what Windows records, never a guess.

The app cannot hear processing that happened before the capture — the signal
it analyses IS the rewritten one. So the note is derived from the OS's own
endpoint record, and every doubt (an unknown spelling, two microphones of one
pair, an inactive endpoint, an unreadable registry) reports nothing at all.
"""

from __future__ import annotations

from v2m.core import driver_fx
from v2m.core.driver_fx import EndpointEffects, endpoint_effects, enhancement_note
from v2m.core.i18n import set_language


def endpoint(
    *,
    name: str = "Microphone",
    adapter: str = "RODE NT-USB",
    state: int = 1,
    disabled: bool = False,
    chain: bool = True,
) -> EndpointEffects:
    return EndpointEffects(name=name, adapter=adapter, state=state, disabled=disabled, chain=chain)


def test_portaudio_spelling_of_one_endpoint_is_identified():
    ends = [endpoint()]
    for spelled in (
        "Microphone (RODE NT-USB)",
        "Microphone (3- RODE NT-USB)",  # the per-run disambiguator
        "Microphone (3- RODE NT-USB) — Windows WASAPI",  # InputDevice.label
    ):
        assert endpoint_effects(spelled, ends) == ends[0]


def test_a_different_adapter_is_a_different_endpoint():
    ends = [endpoint(adapter="Camo")]
    assert endpoint_effects("Microphone (RODE NT-USB)", ends) is None


def test_two_active_endpoints_of_one_pair_are_not_guessed():
    ends = [endpoint(), endpoint()]
    assert endpoint_effects("Microphone (RODE NT-USB)", ends) is None


def test_an_inactive_endpoint_is_not_identified():
    # not present, yet still listed by Windows with an internal flag set
    assert endpoint_effects("Microphone (RODE NT-USB)", [endpoint(state=0x10000004)]) is None


def test_a_chain_the_user_switched_off_is_not_announced():
    assert enhancement_note("Microphone (RODE NT-USB)", [endpoint(disabled=True)]) is None


def test_an_endpoint_without_a_chain_is_not_announced():
    ends = [endpoint(adapter="Camo", chain=False)]
    assert enhancement_note("Microphone (Camo)", ends) is None


def test_the_note_says_what_happens_and_what_to_do():
    ends = [endpoint()]
    set_language("fr")
    french = enhancement_note("Microphone (RODE NT-USB)", ends)
    set_language("en")
    english = enhancement_note("Microphone (RODE NT-USB)", ends)
    try:
        assert french and english and french != english
        assert "améliorations audio" in french
        assert "audio enhancements" in english
    finally:
        set_language("fr")


def test_a_registry_that_cannot_be_read_reports_nothing(monkeypatch):
    monkeypatch.setattr(driver_fx, "_read_endpoints", lambda: [])
    assert enhancement_note("Microphone (RODE NT-USB)") is None

    def denied() -> list[EndpointEffects]:
        raise OSError("access denied")

    monkeypatch.setattr(driver_fx, "_read_endpoints", denied)
    assert enhancement_note("Microphone (RODE NT-USB)") is None


def test_a_null_default_or_a_driver_watermark_is_not_an_effect_chain():
    null_default = {"{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},0": driver_fx._NULL_GUID}
    watermark = {
        "{9e6136e0-57ab-4949-b57a-3627be142855},100": "SWD\\DRIVERENUM\\{x}#VocaEffectPack&1"
    }
    chain = {
        "{d04e05a6-594b-4fb6-a80d-01af5eed7d1d},14": ["{806B324A-6827-46A8-AFD0-8AB09F8E6E61}"]
    }
    assert driver_fx._has_chain(null_default) is False
    assert driver_fx._has_chain(watermark) is False
    assert driver_fx._has_chain(chain) is True


class _FakeKey:
    """A registry key stand-in: its own values plus its sub-keys."""

    def __init__(self, values: dict, subkeys: dict | None = None) -> None:
        self.values = values
        self.subkeys = subkeys or {}

    def __enter__(self) -> "_FakeKey":
        return self

    def __exit__(self, *_: object) -> bool:
        return False


class _FakeRegistry:
    """The four `winreg` calls `driver_fx` makes, over plain dictionaries."""

    HKEY_LOCAL_MACHINE = None

    def OpenKey(self, key: _FakeKey, name: str) -> _FakeKey:
        try:
            return key.subkeys[name]
        except KeyError:
            raise OSError("absent") from None

    def QueryInfoKey(self, key: _FakeKey) -> tuple[int, int, int]:
        return len(key.subkeys), len(key.values), 0

    def EnumValue(self, key: _FakeKey, index: int) -> tuple[str, object, int]:
        name = sorted(key.values)[index]
        return name, key.values[name], 1

    def QueryValueEx(self, key: _FakeKey, name: str) -> tuple[object, int]:
        try:
            return key.values[name], 1
        except KeyError:
            raise OSError("absent") from None


def _endpoint_key(disable_sysfx: dict, effects: dict) -> _FakeKey:
    properties = _FakeKey(
        {
            driver_fx._FRIENDLY_NAME: "Microphone",
            driver_fx._DEVICE_DESCRIPTION: "RODE NT-USB",
            **disable_sysfx,
        }
    )
    return _FakeKey(
        {"DeviceState": 1},
        {"Properties": properties, "FxProperties": _FakeKey(effects)},
    )


def test_the_switch_off_flag_is_read_where_the_driver_keeps_it():
    """Measured: an active RODE NT-USB writes Disable_SysFx into FxProperties."""
    registry = _FakeRegistry()
    in_effects = _endpoint_key({}, {driver_fx._DISABLE_SYSFX: 1})
    in_properties = _endpoint_key({driver_fx._DISABLE_SYSFX: 1}, {})
    assert driver_fx._read_endpoint(registry, in_effects).disabled is True
    assert driver_fx._read_endpoint(registry, in_properties).disabled is True
    assert driver_fx._read_endpoint(registry, _endpoint_key({}, {})).disabled is False


def test_effects_are_active_only_with_a_chain_the_user_left_on():
    assert driver_fx.driver_effects_active("Microphone (RODE NT-USB)", [endpoint()]) is True
    assert (
        driver_fx.driver_effects_active("Microphone (RODE NT-USB)", [endpoint(disabled=True)])
        is False
    )
    assert (
        driver_fx.driver_effects_active(
            "Microphone (Camo)", [endpoint(adapter="Camo", chain=False)]
        )
        is False
    )
    # an endpoint nobody identified never asks for a different capture mode
    assert driver_fx.driver_effects_active("Unknown (Mic)", [endpoint()]) is False


def test_one_endpoint_is_read_from_its_two_sub_keys():
    registry = _FakeRegistry()
    key = _endpoint_key({}, {driver_fx._DISABLE_SYSFX: 0, "{d04e05a6-x},14": ["{A1}"]})
    read = driver_fx._read_endpoint(registry, key)
    assert (read.name, read.adapter, read.state, read.chain) == (
        "Microphone",
        "RODE NT-USB",
        1,
        True,
    )
