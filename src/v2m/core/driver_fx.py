"""What the operating system records about the selected input's processing.

A capture can be filtered BEFORE the app hears it: in shared mode Windows
runs the endpoint's audio effects chain (the sound settings' "audio
enhancements"), so the WAV, the pitch curve, the attacks and the exported
MIDI all describe a signal the driver already rewrote. No evidence INSIDE a
take can reveal that — the evidence IS the signal — so this module reports
what the OS records instead, and says nothing when the endpoint cannot be
identified. Silence is the honest answer there: a guess would either warn
about another microphone's effects or clear this one's.

Windows only, and best-effort by contract: every entry point returns None
off Windows and on any registry failure. This is a diagnostic, never a
reason a capture cannot start, and it never touches the evidence.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any

from .audio_io import INPUT_LABEL_SEPARATOR
from .i18n import tr

# The endpoint store: one sub-key per capture endpoint, `Properties` holding
# the device's identity and `FxProperties` the driver's effect settings.
_CAPTURE_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture"

# PKEY_Device_FriendlyName and PKEY_Device_DeviceDesc: together they spell the
# name PortAudio shows, "friendly (adapter)".
_FRIENDLY_NAME = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"
_DEVICE_DESCRIPTION = "{b3f8fa53-0004-438e-9003-51a46e139bfc},6"
# PKEY_AudioEndpoint_Disable_SysFx: 1 = the user switched the endpoint's audio
# enhancements off. Drivers write it under `Properties` OR under
# `FxProperties` (measured: an active RODE NT-USB keeps it in the latter), so
# reading one of the two reports a switched-off chain as active.
_DISABLE_SYSFX = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},5"
_NULL_GUID = "{00000000-0000-0000-0000-000000000000}"

# DEVICE_STATE_ACTIVE. Windows also lists disabled, not-present and unplugged
# endpoints, which no capture can open; the high bits carry internal flags, so
# only the enum nibble is compared.
_DEVICE_STATE_ACTIVE = 1
_DEVICE_STATE_MASK = 0xF


@dataclass(frozen=True)
class EndpointEffects:
    """What Windows publishes for one capture endpoint."""

    name: str  # friendly name, e.g. "Microphone"
    adapter: str  # device description, e.g. "RODE NT-USB"
    state: int  # DeviceState; the low nibble is the enum
    disabled: bool  # the user switched the endpoint's enhancements off
    chain: bool  # the driver publishes an effects chain


def endpoint_effects(
    device_name: str, endpoints: list[EndpointEffects] | None = None
) -> EndpointEffects | None:
    """The endpoint PortAudio named, or None when it cannot be identified.

    None covers every doubt — an unknown spelling, TWO active endpoints of one
    pair (one interface's two microphones), an inactive endpoint, a registry
    that cannot be read — because both readings of a wrong answer are wrong.
    `endpoints` is the test seam; production reads the registry.
    """
    try:
        pool = _read_endpoints() if endpoints is None else endpoints
    except OSError:
        return None
    wanted = _portaudio_name(device_name)
    found = [endpoint for endpoint in pool if _is_active(endpoint) and _spells(wanted, endpoint)]
    return found[0] if len(found) == 1 else None


def driver_effects_active(device_name: str, endpoints: list[EndpointEffects] | None = None) -> bool:
    """May this endpoint's driver effects change what a capture records?

    True only when Windows publishes an effects chain AND the user has not
    switched it off. Both consumers read this one predicate: the home page
    words it, and the live capture decides whether a raw open is worth
    asking for.
    """
    endpoint = endpoint_effects(device_name, endpoints)
    return bool(endpoint is not None and endpoint.chain and not endpoint.disabled)


def enhancement_note(
    device_name: str, endpoints: list[EndpointEffects] | None = None
) -> str | None:
    """The warning sentence when the driver may process this capture, else None.

    Nothing is said for an endpoint that publishes no effects chain, one the
    user already switched off, or one that cannot be identified: the note
    exists to name a real risk, not to decorate the page.
    """
    if not driver_effects_active(device_name, endpoints):
        return None
    return tr("ui input device effects")


def _portaudio_name(device_name: str) -> str:
    """Drop the host-API suffix `audio_io.InputDevice.label` appends."""
    if INPUT_LABEL_SEPARATOR in device_name:
        return device_name.rsplit(INPUT_LABEL_SEPARATOR, 1)[0]
    return device_name


def _is_active(endpoint: EndpointEffects) -> bool:
    return endpoint.state & _DEVICE_STATE_MASK == _DEVICE_STATE_ACTIVE


def _spells(portaudio_name: str, endpoint: EndpointEffects) -> bool:
    """Does PortAudio's spelling of `endpoint` equal `portaudio_name`?

    PortAudio writes "friendly (adapter)" and inserts an "N- " disambiguator
    before the adapter when Windows offers several endpoints of the same pair
    ("Microphone (3- RODE NT-USB)"). That number is a per-run enumeration
    artifact, not part of the name.
    """
    if not endpoint.name or not endpoint.adapter:
        return False
    pattern = re.escape(f"{endpoint.name} (") + r"(?:\d+- )?" + re.escape(f"{endpoint.adapter})")
    return re.fullmatch(pattern, portaudio_name) is not None


def _has_chain(effects: dict[str, Any]) -> bool:
    """Does the driver publish an effects chain, or only a null default?

    Windows keeps an endpoint's processing under `FxProperties`: an endpoint
    with enhancements carries LISTS of APO GUIDs (the effect objects the
    engine loads), while one that offers none carries a null GUID, nothing at
    all, or a driver watermark. Measured on a Windows 11 desktop: the
    motherboard's Realtek capture endpoint publishes those lists, and its USB
    microphones publish none — only `{9e6136e0-…},100`, a
    "SWD\\DRIVERENUM\\…VocaEffectPack" string that is a device marker, not an
    effect.
    """
    for value in effects.values():
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, str) and item.startswith("{") and item != _NULL_GUID for item in value
        ):
            return True
    return False


def _windows_registry() -> Any | None:
    """The `winreg` module on Windows, None elsewhere (the store is Windows')."""
    if sys.platform != "win32":
        return None
    import winreg

    return winreg


def _read_endpoints() -> list[EndpointEffects]:
    """Every capture endpoint Windows lists (empty off Windows or on failure)."""
    registry = _windows_registry()
    if registry is None:
        return []
    try:
        root = registry.OpenKey(registry.HKEY_LOCAL_MACHINE, _CAPTURE_KEY)
    except OSError:
        return []
    try:
        endpoints: list[EndpointEffects] = []
        for index in range(registry.QueryInfoKey(root)[0]):
            try:
                with registry.OpenKey(root, registry.EnumKey(root, index)) as key:
                    endpoints.append(_read_endpoint(registry, key))
            except OSError:
                continue  # one unreadable endpoint must not hide the others
        return endpoints
    finally:
        root.Close()


def _read_endpoint(registry: Any, key: Any) -> EndpointEffects:
    """One endpoint's identity, state and effect settings."""
    properties = _values(registry, key, "Properties")
    effects = _values(registry, key, "FxProperties")
    try:
        state = int(registry.QueryValueEx(key, "DeviceState")[0])
    except OSError:
        state = 0
    disabled = properties.get(_DISABLE_SYSFX, effects.get(_DISABLE_SYSFX, 0)) == 1
    return EndpointEffects(
        name=str(properties.get(_FRIENDLY_NAME, "")),
        adapter=str(properties.get(_DEVICE_DESCRIPTION, "")),
        state=state,
        disabled=disabled,
        chain=_has_chain(effects),
    )


def _values(registry: Any, key: Any, subkey: str) -> dict[str, Any]:
    """Every value under one endpoint sub-key (an absent sub-key reads empty).

    `registry` and `key` stay untyped: the four calls this module makes are the
    whole contract, and the tests drive them with a dictionary-backed stand-in.
    """
    out: dict[str, Any] = {}
    try:
        with registry.OpenKey(key, subkey) as handle:
            for index in range(registry.QueryInfoKey(handle)[1]):
                name, value, _kind = registry.EnumValue(handle, index)
                out[str(name)] = value
    except OSError:
        pass
    return out
