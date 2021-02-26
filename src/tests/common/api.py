import unittest


class AbstractTestCase:
    test: unittest.TestCase

    def setUp(self):
        self.test = self
