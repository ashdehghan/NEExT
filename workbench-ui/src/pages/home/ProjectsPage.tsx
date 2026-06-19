import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FilePlus2, RotateCcw, Trash2 } from "lucide-react";
import { api, type ProjectManifest } from "../../api";
import { ConfirmDialog } from "../../components/primitives/ConfirmDialog";
import { EmptyState } from "../../components/primitives/EmptyState";
import { FcIcon } from "../../components/primitives/FcIcon";

interface CreateProjectViewProps {
  onCreated: (id: string) => void;
}

interface ProjectsViewProps {
  workspacePath: string;
  projects: ProjectManifest[];
  activeProjectId: string;
  onSelectProject: (id: string) => void;
}

export function CreateProjectView({ onCreated }: CreateProjectViewProps) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const createProject = useMutation({
    mutationFn: () => api.createProject({ name: name.trim(), description: description.trim() }),
    onSuccess: (project) => {
      setName("");
      setDescription("");
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["workspace"] });
      onCreated(project.id);
    }
  });

  return (
    <div className="workflow">
      <section className="card" id="form-create-project">
        <header className="card-head">
          <span className="card-head-fc">
            <FcIcon name="projects" size={32} />
          </span>
          <h3>Create Project</h3>
        </header>
        <div className="card-body">
          <div className="field-grid">
            <label className="field field-wide">
              <span>Name</span>
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                data-form-first-input
              />
            </label>
            <label className="field field-wide">
              <span>Description</span>
              <textarea
                rows={4}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </label>
          </div>
          {createProject.error && <p className="error-text">{createProject.error.message}</p>}
        </div>
        <footer className="card-foot">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => createProject.mutate()}
            disabled={createProject.isPending || !name.trim()}
          >
            <FilePlus2 /> {createProject.isPending ? "Creating" : "Create"}
          </button>
        </footer>
      </section>
    </div>
  );
}

export function ProjectsView({
  workspacePath,
  projects,
  activeProjectId,
  onSelectProject
}: ProjectsViewProps) {
  const queryClient = useQueryClient();
  const [projectToDelete, setProjectToDelete] = useState<ProjectManifest | null>(null);

  const deleteProject = useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: (summary) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["workspace"] });
      if (summary.id === activeProjectId) {
        onSelectProject("");
      }
      setProjectToDelete(null);
    }
  });

  const confirmDelete = () => {
    if (projectToDelete) {
      deleteProject.mutate(projectToDelete.id);
    }
  };

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            Projects · {projects.length} {projects.length === 1 ? "project" : "projects"}
          </span>
          <span className="muted mono">{workspacePath}</span>
        </header>
        {projects.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>No projects.</EmptyState>
          </div>
        ) : (
          <div className="artifact-table-scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Description</th>
                  <th>Updated</th>
                  <th>Status</th>
                  <th className="actions-col">Actions</th>
                </tr>
              </thead>
              <tbody>
                {projects.map((project) => (
                  <tr
                    key={project.id}
                    className={project.id === activeProjectId ? "is-selected" : ""}
                    onClick={() => onSelectProject(project.id)}
                  >
                    <td>
                      <strong>{project.name}</strong>
                    </td>
                    <td className="muted">{project.description || ""}</td>
                    <td className="muted">{project.updated_at}</td>
                    <td>
                      <span className={`status-pill ${project.id === activeProjectId ? "is-ready" : "is-idle"}`}>
                        {project.id === activeProjectId ? "active" : "idle"}
                      </span>
                    </td>
                    <td className="actions-cell">
                      <button
                        type="button"
                        className="icon-btn icon-btn-danger"
                        aria-label={`Delete ${project.name}`}
                        title={`Delete ${project.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          deleteProject.reset();
                          setProjectToDelete(project);
                        }}
                      >
                        <Trash2 />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      {projectToDelete ? (
        <ConfirmDialog
          title="Delete Project"
          message={`Move "${projectToDelete.name}" to workspace trash?`}
          confirmLabel="Delete project"
          busy={deleteProject.isPending}
          error={deleteProject.error?.message}
          onCancel={() => {
            if (!deleteProject.isPending) {
              deleteProject.reset();
              setProjectToDelete(null);
            }
          }}
          onConfirm={confirmDelete}
        />
      ) : null}
    </div>
  );
}

export function TrashView() {
  const queryClient = useQueryClient();
  const trash = useQuery({ queryKey: ["trash"], queryFn: api.trash });

  const invalidateAfterRestore = () => {
    queryClient.invalidateQueries({ queryKey: ["trash"] });
    queryClient.invalidateQueries({ queryKey: ["workspace"] });
    queryClient.invalidateQueries({ queryKey: ["projects"] });
  };

  const restoreProject = useMutation({
    mutationFn: (trashId: string) => api.restoreProject(trashId),
    onSuccess: invalidateAfterRestore
  });
  const restoreArtifactDeletion = useMutation({
    mutationFn: ({ projectId, bundleId }: { projectId: string; bundleId: string }) => api.restoreArtifactDeletion(projectId, bundleId),
    onSuccess: invalidateAfterRestore
  });

  const projects = trash.data?.projects || [];
  const bundles = trash.data?.artifact_deletions || [];

  return (
    <div className="workflow">
      <section className="artifact-table">
        <header className="artifact-table-head">
          <span className="artifact-table-title">
            <FcIcon name="trash" size={16} />
            Trash · {projects.length + bundles.length} {projects.length + bundles.length === 1 ? "item" : "items"}
          </span>
          <span className="muted">Restore moves items back to their original IDs.</span>
        </header>
        {trash.error ? <p className="table-error">{trash.error.message}</p> : null}
        {restoreProject.error ? <p className="table-error">{restoreProject.error.message}</p> : null}
        {restoreArtifactDeletion.error ? <p className="table-error">{restoreArtifactDeletion.error.message}</p> : null}
        {trash.isLoading ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Loading trash.</EmptyState>
          </div>
        ) : projects.length === 0 && bundles.length === 0 ? (
          <div className="artifact-table-empty">
            <EmptyState compact>Trash is empty.</EmptyState>
          </div>
        ) : (
          <div className="trash-stack">
            <div className="trash-section">
              <h3>Projects</h3>
              {projects.length === 0 ? (
                <p className="muted">No trashed projects.</p>
              ) : (
                <div className="artifact-table-scroll">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Description</th>
                        <th>Trash Path</th>
                        <th className="actions-col">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {projects.map((project) => (
                        <tr key={project.trash_id}>
                          <td>
                            <strong>{project.name}</strong>
                          </td>
                          <td className="muted">{project.description}</td>
                          <td className="muted mono">{project.trashed_path}</td>
                          <td className="actions-cell">
                            <button
                              type="button"
                              className="btn"
                              disabled={restoreProject.isPending}
                              onClick={() => restoreProject.mutate(project.trash_id)}
                            >
                              <RotateCcw />
                              Restore
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            <div className="trash-section">
              <h3>Artifact Delete Bundles</h3>
              {bundles.length === 0 ? (
                <p className="muted">No trashed artifact bundles.</p>
              ) : (
                <div className="artifact-table-scroll">
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Root</th>
                        <th>Mode</th>
                        <th>Artifacts</th>
                        <th>Created</th>
                        <th className="actions-col">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bundles.map((bundle) => (
                        <tr key={bundle.id}>
                          <td>
                            <strong>{bundle.root_artifact.name}</strong>
                            <span className="muted"> · {bundle.root_artifact.artifact_kind}</span>
                          </td>
                          <td>
                            <span className="status-pill is-idle">{bundle.delete_mode}</span>
                          </td>
                          <td>{bundle.artifacts.length}</td>
                          <td className="muted mono">{bundle.created_at}</td>
                          <td className="actions-cell">
                            <button
                              type="button"
                              className="btn"
                              disabled={restoreArtifactDeletion.isPending}
                              onClick={() => restoreArtifactDeletion.mutate({ projectId: bundle.project_id, bundleId: bundle.id })}
                            >
                              <RotateCcw />
                              Restore
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
