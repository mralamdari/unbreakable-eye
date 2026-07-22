# Security Policy

## Supported Versions

We currently support the latest release with security updates.

| Version | Supported |
|---------|-----------|
| latest  | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability in Unbreakable Eye, please **do not** open a public issue.

Email: [mralamdari2000@gmail.com](mailto:mralamdari2000@gmail.com)

We will acknowledge receipt within 48 hours and provide a timeline for a fix. Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We appreciate responsible disclosure and will credit you in the release notes.

## Scope

This policy covers:
- The Python application under `src/`
- The Cloudflare Worker under `cloudflare-worker/`
- Docker configurations under `infra/`
- Nginx configuration under `infra/nginx/`

Database credentials, secret keys, and Telegram tokens should **never** be committed to the repository.
