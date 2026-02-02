import tempfile
import unittest
from unittest.mock import patch

from mia import parse_results


class TestParseResults(unittest.TestCase):
    def test_main_calls_plot_average(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(parse_results.evaluation, "plot_average_roc_curves") as plot_fn:
                parse_results.main(["--resdir", tmpdir, "--low-exponent", "-3"])
            plot_fn.assert_called_once_with(tmpdir, low_exponent=-3)


if __name__ == "__main__":
    unittest.main()
