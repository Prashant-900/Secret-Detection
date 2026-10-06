"""Explicit coverage axes for synthetic data; these are targets, not measured gaps.

Use ``audit`` on a local CredData checkout before assigning quotas. Synthetic test
sets are stress tests, never substitutes for repository-held-out real examples.
"""

FAMILIES = {
    "terraform": {"language": "hcl", "path": "main.tf", "topics": ["database provisioning", "container runtime", "observability provider", "service account configuration"]},
    "kubernetes": {"language": "yaml", "path": "deployment.yaml", "topics": ["application deployment", "batch worker", "database migration", "metrics collector"]},
    "helm": {"language": "yaml", "path": "values.yaml", "topics": ["web application values", "data platform values", "monitoring values", "registry values"]},
    "ansible": {"language": "yaml", "path": "group_vars/application.yml", "topics": ["database users", "application configuration", "service deployment", "backup settings"]},
    "compose": {"language": "yaml", "path": "compose.yaml", "topics": ["web and database stack", "background jobs", "collaboration server", "local monitoring"]},
    "github_actions": {"language": "yaml", "path": ".github/workflows/build.yml", "topics": ["release publication", "container build", "integration tests", "deployment workflow"]},
    "cloudformation": {"language": "json", "path": "template.json", "topics": ["database stack", "application service", "queue worker", "deployment parameters"]},
    "dotenv": {"language": "dotenv", "path": ".env", "topics": ["mail configuration", "database configuration", "payment integration", "web session configuration"]},
    "toml": {"language": "toml", "path": "config.toml", "topics": ["Rust application settings", "package registry", "data connector", "telemetry exporter"]},
    "json_config": {"language": "json", "path": "config.json", "topics": ["service settings", "database connector", "webhook integration", "application deployment"]},
    "ini": {"language": "ini", "path": "settings.ini", "topics": ["Python application settings", "database DSN", "worker configuration", "mail relay"]},
    "xml": {"language": "xml", "path": "application.xml", "topics": ["Spring datasource", "Maven server settings", "application realm", "message broker"]},
    "python": {"language": "python", "path": "settings.py", "topics": ["web application settings", "database client", "SDK initialization", "background worker"]},
    "javascript": {"language": "javascript", "path": "config.js", "topics": ["Node service settings", "browser and server configuration", "database client", "SDK initialization"]},
    "java": {"language": "java", "path": "Configuration.java", "topics": ["JDBC configuration", "HTTP client setup", "service client", "mail configuration"]},
    "go": {"language": "go", "path": "config.go", "topics": ["database connection", "API client configuration", "service initialization", "worker configuration"]},
    "jenkins": {"language": "groovy", "path": "Jenkinsfile", "topics": ["release pipeline", "container publication", "deployment credentials", "integration pipeline"]},
    "pulumi": {"language": "typescript", "path": "index.ts", "topics": ["database deployment", "service configuration", "container workload", "provider configuration"]},
}

SECRET_TYPES = (
    "password_low_entropy", "password_random", "opaque_token", "api_key_hex",
    "github_token", "aws_access_key", "aws_secret_key", "jwt",
    "connection_string", "private_key_pem", "base64_secret",
    "stripe_secret", "slack_bot_token", "gitlab_token", "huggingface_token",
    "npm_token", "google_api_key", "azure_client_secret", "basic_auth",
)
LOCALES = ("en", "es", "hi", "fr", "de")
KEY_HINTS = ("explicit", "neutral", "misleading")
STYLES = ("minimal", "nested", "legacy", "commented", "multi_environment", "sdk_call")
DEFAULT_OOD_FAMILIES = ("jenkins", "pulumi")

def category_for(secret_type):
    if secret_type == "password_low_entropy":
        return "low_entropy_secret"
    if secret_type in ("github_token", "aws_access_key", "aws_secret_key", "jwt", "private_key_pem",
                       "stripe_secret", "slack_bot_token", "gitlab_token", "huggingface_token",
                       "npm_token", "google_api_key", "basic_auth"):
        return "known_format_secret"
    if secret_type == "base64_secret":
        return "encoded_secret"
    return "unknown_format_secret"
