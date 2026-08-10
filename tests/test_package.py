import unittest

from neural_cutting_stock import __version__


class PackageMetadataTests(unittest.TestCase):
    def test_package_version_is_exposed(self) -> None:
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
