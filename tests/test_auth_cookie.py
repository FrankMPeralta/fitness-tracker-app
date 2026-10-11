import json
import unittest

from app import AUTH_COOKIE_NAME, load_auth_cookie


class Cookies:
    def __init__(self, value):
        self.value = value
        self.deleted = False

    def get(self, name):
        assert name == AUTH_COOKIE_NAME
        return self.value

    def delete(self, name):
        assert name == AUTH_COOKIE_NAME
        self.deleted = True


class AuthCookieTests(unittest.TestCase):
    def test_decoded_cookie_from_browser_is_preserved(self):
        session = {'access_token': 'test-access', 'refresh_token': 'test-refresh'}
        cookies = Cookies(session)
        self.assertEqual(load_auth_cookie(cookies), session)
        self.assertFalse(cookies.deleted)

    def test_string_cookie_is_supported(self):
        session = {'access_token': 'test-access', 'refresh_token': 'test-refresh'}
        cookies = Cookies(json.dumps(session))
        self.assertEqual(load_auth_cookie(cookies), session)
        self.assertFalse(cookies.deleted)

    def test_invalid_cookie_shapes_are_removed(self):
        for value in ('[1]', 'null', '42', 'invalid-json', {'access_token': 123}, ['token']):
            with self.subTest(value=value):
                cookies = Cookies(value)
                self.assertIsNone(load_auth_cookie(cookies))
                self.assertTrue(cookies.deleted)

    def test_unavailable_or_empty_cookie(self):
        self.assertIsNone(load_auth_cookie(None))
        cookies = Cookies(None)
        self.assertIsNone(load_auth_cookie(cookies))
        self.assertFalse(cookies.deleted)
