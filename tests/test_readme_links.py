"""README.md is the PyPI project page, and that is a different document.

WHY THIS FILE EXISTS
--------------------
0.1.0 shipped with all eleven of its documentation links broken on PyPI, and
nobody could have noticed by reading the repository. A link written
``[guide](docs/getting-started.md)`` renders and resolves perfectly on GitHub,
because GitHub resolves it against the repository. PyPI serves the same text at
``pypi.org/project/focusedgaze/``, where it resolves against ``pypi.org`` and
404s. The two renderings disagree, the correct-looking one is the one a
developer sees, and the broken one is the one users get.

That is not a mistake anybody catches by proofreading. It needs a test, and the
test has to encode the rule rather than the symptom: **no relative link may
appear in README.md**, whatever it points at.

The links were corrected in ``8a079bf``, after the ``v0.1.0`` tag. PyPI refuses
re-uploads of a version, so 0.1.0's page stays broken permanently and 0.1.1 is
what delivers the fix. This test is what stops a third occurrence.

WHAT IS *NOT* CHECKED HERE
--------------------------
Whether the URLs resolve over the network. That would make the suite fail on an
aeroplane, and fail for a contributor whose branch legitimately references a
document not yet on ``main``. What is checkable offline is the shape of the link
and the existence of the file it names inside this repository, and both of those
are what actually went wrong.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: ``[text](target)``. Deliberately not a full Markdown parser: this runs
#: against one file whose link syntax is known and plain.
_LINK = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: The repository, as the links must spell it.
_REPO = "https://github.com/muhammad-asifkhan/focusedgaze"


def _links() -> list[tuple[str, str]]:
    text = README.read_text(encoding="utf-8")
    return [(m.group("text"), m.group("target")) for m in _LINK.finditer(text)]


def test_the_readme_exists_and_has_links() -> None:
    """Guards the guard: a regex that silently matched nothing would make every
    assertion below vacuously true."""
    assert README.is_file()
    assert len(_links()) >= 5, "the link regex matched almost nothing; check it"


def test_no_link_is_repository_relative() -> None:
    """THE test. A relative link works on GitHub and 404s on PyPI.

    In-page anchors (``#licence``) are exempt: they resolve within the rendered
    description itself and work identically in both places.
    """
    relative = [
        (text, target)
        for text, target in _links()
        if not target.startswith(("http://", "https://", "#", "mailto:"))
    ]
    assert not relative, (
        "README.md has links that resolve against pypi.org and 404 there:\n"
        + "\n".join(f"  [{t}]({u})" for t, u in relative)
        + f"\nWrite them absolute: {_REPO}/blob/main/<path>"
    )


def test_every_repository_link_names_a_file_that_exists() -> None:
    """The other half. Absolute is necessary and not sufficient: an absolute URL
    to a document nobody ever wrote is still a 404, and this catches a rename
    without needing the network."""
    missing: list[str] = []
    prefix = f"{_REPO}/blob/"
    for _text, target in _links():
        if not target.startswith(prefix):
            continue
        # <prefix><ref>/<path> -- drop the ref, keep the repository path.
        remainder = target[len(prefix):]
        if "/" not in remainder:
            continue
        _ref, path = remainder.split("/", 1)
        path = path.split("#", 1)[0]
        if not (ROOT / path).exists():
            missing.append(f"{path}  (from {target})")
    assert not missing, "README links name files that are not in this repository:\n" + "\n".join(
        f"  {m}" for m in missing
    )


def test_the_headline_guide_is_linked() -> None:
    """`getting-started.md` is the document the README calls "Start here", and
    the one whose 404 cost the most on the 0.1.0 page."""
    targets = [target for _text, target in _links()]
    assert any("docs/getting-started.md" in t for t in targets)


@pytest.mark.parametrize("url_name", ["Homepage", "Documentation", "Changelog", "Source", "Issues"])
def test_the_pypi_sidebar_links_are_declared(url_name: str) -> None:
    """``[project.urls]`` renders as the PyPI sidebar. Those links do not depend
    on the description rendering at all, which makes them the one part of the
    page that cannot be broken by a Markdown difference."""
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        pytest.skip("tomllib needs Python 3.11+")

    with (ROOT / "pyproject.toml").open("rb") as handle:
        urls = tomllib.load(handle)["project"]["urls"]
    assert url_name in urls, f"{url_name} is missing from [project.urls]"
    assert urls[url_name].startswith("https://"), f"{url_name} is not an absolute https URL"


def test_the_version_is_single_sourced() -> None:
    """The link fix only reaches PyPI through a new version: uploads of an
    existing one are refused. So a release that carries a README change and
    forgets the bump silently ships nothing."""
    from focusedgaze import __version__

    with (ROOT / "pyproject.toml").open("rb") as handle:
        import tomllib

        project = tomllib.load(handle)["project"]
    assert "version" in project.get("dynamic", []), (
        "the version must stay dynamic, read from __init__ by hatch (D6)"
    )
    assert __version__ != "0.1.0", (
        "0.1.0 is published with broken PyPI links and cannot be re-uploaded; "
        "shipping the fix needs a new version"
    )
