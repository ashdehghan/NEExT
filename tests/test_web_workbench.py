import pandas as pd

from NEExT.cli import main
from NEExT.web.jobs import JobManager
from NEExT.web.project import ProjectManager


def test_cli_help_path_does_not_require_web_dependencies(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "NEExT command line tools" in captured.out


def test_project_manager_registers_and_loads_table_and_object(tmp_path):
    project = ProjectManager(tmp_path)
    df = pd.DataFrame({"graph_id": [0, 1], "emb_0": [0.1, 0.2]})
    obj = {"kind": "embeddings"}

    artifact = project.register_artifact(
        artifact_type="embeddings",
        name="demo embeddings",
        obj=obj,
        dataframe=df,
        metadata={"embedding_algorithm": "approx_wasserstein"},
        parent_ids=["dataset-1"],
    )

    manifest = project.manifest()
    assert manifest["schema_version"] == 1
    assert manifest["artifacts"][0]["id"] == artifact.id
    assert manifest["artifacts"][0]["metadata"]["embedding_algorithm"] == "approx_wasserstein"
    assert project.load_object(artifact.id) == obj
    pd.testing.assert_frame_equal(project.load_table(artifact.id), df)


def test_job_manager_persists_successful_job(tmp_path):
    project = ProjectManager(tmp_path)
    jobs = JobManager(project, max_workers=1)

    def work(update):
        update(0.5, "halfway")
        artifact = project.register_artifact("dataset", "demo", obj={"ok": True})
        return artifact.id

    record = jobs.submit("dataset", "create demo", work)
    jobs.futures[record.id].result(timeout=10)

    completed = jobs.get(record.id)
    assert completed.status == "succeeded"
    assert completed.progress == 1.0
    assert completed.result_artifact_id is not None
    assert any(job["id"] == record.id and job["status"] == "succeeded" for job in project.list_jobs())
