from pathlib import Path
from subprocess import check_output

from conftest import GitMinirepo


def test_minirepo_history(minirepo_git: GitMinirepo) -> None:
    root, base, head = minirepo_git

    def git(*args: str) -> str:
        return check_output(["git", *args], cwd=root, text=True).strip()

    assert git("rev-parse", "HEAD^") == base
    assert git("rev-parse", "HEAD") == head
    assert git("diff", "--name-only", base, head) == "postbag/express.py"
    assert '"X1"' in git("show", f"{base}:postbag/express.py")
    assert '"X2"' in git("show", f"{head}:postbag/express.py")
    assert git("status", "--porcelain") == ""
    assert (root / "postbag" / "empty.py").read_bytes() == b""
    assert not (Path(__file__).parent / "fixtures" / "minirepo" / ".git").exists()
