"""
SAGA Init CLI Command
=====================

Entrypoint for initializing new SAGA projects.
Wraps saga.core.onboarding.OnboardingService.
"""

import argparse
import sys
from pathlib import Path

from saga.core.onboarding import OnboardingConfig, OnboardingService


def run_init(args: argparse.Namespace) -> int:
    """Execute the onboarding flow."""
    root = Path.cwd()

    # Configure onboarding
    config = OnboardingConfig(
        project_root=root,
        non_interactive=args.non_interactive,
        profile=args.profile,
        strictness=args.strictness,
        force=args.yes
    )

    # Initialize service
    service = OnboardingService(config)

    try:
        # Run onboarding
        result = service.run()

        # Report results
        print(f"\n✅ SAGA Initialized in {root}")

        if result.created:
            print("\nCreated:")
            for p in result.created:
                print(f"  + {p.relative_to(root)}")

        if result.modified:
            print("\nModified:")
            for p in result.modified:
                print(f"  ~ {p.relative_to(root)}")

        if result.skipped:
            print(f"\nSkipped {len(result.skipped)} existing files (use --yes to force)")

        print("\nNext steps:")
        print("1. Run `saga serve` to start the server")
        print("2. Check .saga/config.yaml for governance settings")

        return 0

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Initialize SAGA in the current project.")

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompting (requires --profile)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Explicitly set profile (e.g. python-fastapi)"
    )
    parser.add_argument(
        "--strictness",
        choices=["relaxed", "standard", "faang"],
        default="standard",
        help="Set governance strictness level"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmations and overwrite non-user files"
    )

    args = parser.parse_args()
    sys.exit(run_init(args))


if __name__ == "__main__":
    main()
