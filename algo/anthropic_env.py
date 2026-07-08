from __future__ import annotations

import os

SUPPORTED_ANTHROPIC_PROFILES = frozenset({"COMPASS"})


def _read_non_empty_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def apply_selected_anthropic_profile_to_environment() -> str | None:
    profile = _read_non_empty_env("ANTHROPIC_PROFILE")
    if profile is None:
        return None

    normalized_profile = profile.upper()
    if normalized_profile not in SUPPORTED_ANTHROPIC_PROFILES:
        supported_profiles = ", ".join(sorted(SUPPORTED_ANTHROPIC_PROFILES))
        raise ValueError(
            f"ANTHROPIC_PROFILE must be one of [{supported_profiles}], got {profile!r}."
        )

    token_env = f"ANTHROPIC_{normalized_profile}_AUTH_TOKEN"
    base_url_env = f"ANTHROPIC_{normalized_profile}_BASE_URL"
    token = _read_non_empty_env(token_env)
    base_url = _read_non_empty_env(base_url_env)

    missing_envs: list[str] = []
    if token is None:
        missing_envs.append(token_env)
    if base_url is None:
        missing_envs.append(base_url_env)
    if missing_envs:
        missing_names = ", ".join(missing_envs)
        raise ValueError(
            f"ANTHROPIC_PROFILE={normalized_profile} requires non-empty values for [{missing_names}]."
        )

    os.environ["ANTHROPIC_PROFILE"] = normalized_profile
    os.environ["ANTHROPIC_AUTH_TOKEN"] = token
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    return normalized_profile
