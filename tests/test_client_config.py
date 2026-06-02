import subprocess
import sys
import unittest


class ClientConfigTests(unittest.TestCase):
    def test_client_config_prints_without_server(self):
        result = subprocess.run(
            [
                sys.executable,
                "client.py",
                "--config",
                "--host",
                "127.0.0.1",
                "--port",
                "9",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = result.stdout + result.stderr
        self.assertIn("AlphaZero Actor Config", output)
        self.assertNotIn("Waiting for server", output)


if __name__ == "__main__":
    unittest.main()
