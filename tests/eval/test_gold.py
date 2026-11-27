from pathlib import Path

import pytest

from reposcout.eval.gold import extract_gold

FIXTURE = Path(__file__).parents[1] / "fixtures" / "minirepo"


@pytest.mark.parametrize(
    ("path", "hunk", "functions"),
    [
        ("postbag/zones.py", "@@ -3,0 +4 @@\n+        return value\n", ["destination_zone.prefix"]),
        (
            "postbag/zones.py",
            "@@ -2,2 +2,3 @@\n     def prefix(value: str) -> str:\n"
            "+        value = value.lower()\n         return value.strip()\n",
            ["destination_zone.prefix"],
        ),
        (
            "postbag/rates.py",
            "@@ -10 +10 @@\n-        if grams < self.letter_limit_grams:\n"
            "+        if grams <= self.letter_limit_grams:\n@@ -16 +16 @@\n-        base = 120\n"
            "+        base = 121\n",
            ["RateCard.quote", "RateCard.rate_band"],
        ),
        ("postbag/returns.py", "@@ -6 +6 @@\n-@cache\n+@staticmethod\n", ["label_prefix"]),
        (
            "postbag/rates.py",
            "@@ -6 +6 @@\n-    letter_limit_grams = 500\n+    letter_limit_grams = 501\n",
            [],
        ),
        ("postbag/express.py", '@@ -3 +3 @@\n-SERVICE_CODE = "X1"\n+SERVICE_CODE = "X2"\n', []),
        ("postbag/express.py", "@@ -0,0 +1 @@\n+# header\n", []),
    ],
)
def test_preimage_mapping(path: str, hunk: str, functions: list[str]) -> None:
    gold = extract_gold(f"--- a/{path}\n+++ b/{path}\n{hunk}", FIXTURE)
    assert gold.files == [path]
    assert gold.functions == [f"{path}::{symbol}" for symbol in functions]


def test_deleted_and_renamed_files_use_old_paths_and_new_file_is_ignored() -> None:
    patch = """diff --git a/postbag/zones.py b/postbag/areas.py
similarity index 100%
rename from postbag/zones.py
rename to postbag/areas.py
diff --git a/postbag/rates.py b/postbag/rates.py
deleted file mode 100644
--- a/postbag/rates.py
+++ /dev/null
@@ -10 +0,0 @@
-        if grams < self.letter_limit_grams:
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1 @@
+print('new')
"""
    gold = extract_gold(patch, FIXTURE)
    assert gold.files == ["postbag/rates.py", "postbag/zones.py"]
    assert gold.functions == ["postbag/rates.py::RateCard.rate_band"]


def test_patch_cannot_read_outside_checkout(tmp_path: Path) -> None:
    (tmp_path / "link.py").symlink_to(FIXTURE / "postbag" / "zones.py")
    with pytest.raises(ValueError, match="boundary"):
        extract_gold("--- a/link.py\n+++ b/link.py\n@@ -1 +1 @@\n-old\n+new\n", tmp_path)
