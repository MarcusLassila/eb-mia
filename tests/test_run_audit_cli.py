import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from mia import run_audit as run_audit_module


class TestRunAuditCli(unittest.TestCase):
    def _write_config(self, data):
        handle, path = tempfile.mkstemp(suffix=".yaml")
        os.close(handle)
        with open(path, "w") as file:
            yaml.safe_dump(data, file)
        return path

    def test_main_calls_sample_audit(self):
        path = self._write_config({"audit_mode": "sample"})
        with (
            patch.object(run_audit_module, "run_audit") as run_audit_fn,
            patch.object(run_audit_module, "run_entity_audit") as run_entity_fn,
            patch.object(run_audit_module.torch.cuda, "is_available", return_value=False),
        ):
            run_audit_module.main(["--config", path])
        run_audit_fn.assert_called_once()
        run_entity_fn.assert_not_called()
        os.remove(path)

    def test_main_calls_entity_audit(self):
        path = self._write_config({"audit_mode": "entity"})
        with (
            patch.object(run_audit_module, "run_audit") as run_audit_fn,
            patch.object(run_audit_module, "run_entity_audit") as run_entity_fn,
            patch.object(run_audit_module.torch.cuda, "is_available", return_value=False),
        ):
            run_audit_module.main(["--config", path])
        run_entity_fn.assert_called_once()
        run_audit_fn.assert_not_called()
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
