# ASGI entrypoint wrapper so uvicorn can find the FastAPI app
# This uses a lazy proxy so importing this module doesn't eagerly import
# `api.main` (which may run heavy initialization). Uvicorn can import
# `asgi:app` safely and the real FastAPI app is loaded on first request.
#
# Usage: uvicorn asgi:app --reload --host 0.0.0.0 --port 8000

from typing import Optional


class _LazyApp:
	"""ASGI callable that lazily imports and delegates to the real app.

	This avoids import-time side-effects when uvicorn imports this module.
	"""
	def __init__(self):
		self._app = None

	def _load(self):
		if self._app is None:
			# Import here to avoid running api.main at module import time
			from importlib import import_module
			m = import_module('api.main')
			self._app = getattr(m, 'app')
		return self._app

	def __call__(self, scope, receive, send):
		app = self._load()
		return app(scope, receive, send)


# Expose an ASGI callable named `app` for uvicorn: `uvicorn asgi:app`
app = _LazyApp()
