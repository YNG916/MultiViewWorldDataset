from multi_view_world_dataset.cli import build_parser


def test_generate_command_is_explicit_and_large_generation_is_opt_in():
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--config",
            "configs/smoke.yaml",
            "--scene",
            "Rs_int",
            "--output-root",
            "/tmp/mvwd-test-output",
        ]
    )
    assert args.command == "generate"
    assert args.scene == "Rs_int"
    assert args.allow_large is False
