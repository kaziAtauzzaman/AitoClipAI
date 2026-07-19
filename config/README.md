# config

Configuration files and templates will live in this directory.

Keep environment-specific secrets out of version control and prefer documented
example files for shared settings.

For YouTube uploads, copy `youtube-upload.example.json` to the ignored
`youtube-upload.json` and point it at a Google OAuth desktop-client secrets
file. Paths can instead be supplied with:

- `AITOCLIP_YOUTUBE_CLIENT_SECRETS_PATH`
- `AITOCLIP_YOUTUBE_TOKEN_PATH`
- `AITOCLIP_UPLOAD_LEDGER_PATH`

The client secrets, generated refresh token, and runtime ledger must never be
committed.
