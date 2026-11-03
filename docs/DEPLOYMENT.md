# Deployment preparation

The public repository and GitHub Release checkpoints allow a CPU Streamlit deployment. The documented local-run guide satisfies the application-access requirement.

Streamlit Community Cloud deployment requires a signed-in Streamlit account linked to GitHub. GitHub CLI credentials alone do not create a Community Cloud session. No authenticated Streamlit deployment session is available through the connected tools. Public deployment is therefore NOT VERIFIED; no public app URL has been assigned.

If deploying from an authenticated account: choose the public repository `commedeschamps/WeatherVision`, branch `main`, entrypoint `app.py`, Python 3.12. Install requirements.txt, then run `python scripts/download_checkpoints.py` during a supported setup/start command. The app itself never trains and does not download weights implicitly. Hosts without a pre-start command can use `deploy_app.py` as the entrypoint; it verifies/downloads released app weights once before loading app.py.

The three released app checkpoints occupy approximately 115 MB on disk; loaded tensors and dependencies require additional RAM. Resource use must be measured on the chosen host. Do not assume the nominal file size equals process memory. Limit concurrent model work, keep CPU inference, and test upload, all three models, Grad-CAM, comparison and downloads after deployment. Dataset download is optional for example browsing.

References checked 2026-11-03: [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [resource management](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
