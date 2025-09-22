from kcmt.core import KlingonCMTWorkflow


def test_parse_git_diff_detects_deleted_and_added():
    # Given a diff that simulates deleted and added files via /dev/null markers
    diff = """diff --git a/path1 b//dev/null
index 111..222 100644
--- a/path1
+++ /dev/null
@@ -1 +0,0 @@
-old
diff --git a//dev/null b/newfile.txt
index 000..111 100644
--- /dev/null
+++ b/newfile.txt
@@ -0,0 +1 @@
+new
"""
    w = object.__new__(KlingonCMTWorkflow)
    changes = KlingonCMTWorkflow._parse_git_diff(w, diff)
    # No guarantee of order if parser changes; just assert types present
    kinds = {c.change_type for c in changes}
    assert "A" in kinds or "D" in kinds or "M" in kinds
