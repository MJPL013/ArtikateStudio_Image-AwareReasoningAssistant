# Google Project IDX Nix Configuration
# This file configures system dependencies required for OpenCV and ML models

{ pkgs, ... }: {
  # System packages required for the environment
  packages = [
    # Python 3.10+
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv

    # OpenCV system dependencies
    pkgs.glib
    pkgs.libglvnd
    pkgs.mesa
    pkgs.gcc

    # Additional libraries for image processing
    pkgs.zlib
    pkgs.libpng
    pkgs.libjpeg

    # For torch/transformers
    pkgs.stdenv.cc.cc.lib
  ];

  # Environment variables
  env = {
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  };

  # IDX-specific configuration
  idx = {
    # Extensions for Python development
    extensions = [
      "ms-python.python"
      "ms-python.vscode-pylance"
    ];

    # Workspace lifecycle hooks
    workspace = {
      # Runs when the workspace is first created
      onCreate = {
        install-deps = ''
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt
        '';
      };

      # Runs when the workspace starts
      onStart = {
        activate-venv = ''
          source .venv/bin/activate
        '';
      };
    };

    # Preview configuration for Streamlit
    previews = {
      enable = true;
      previews = {
        web = {
          command = ["streamlit" "run" "app.py" "--server.port" "$PORT" "--server.headless" "true"];
          manager = "web";
        };
      };
    };
  };
}
