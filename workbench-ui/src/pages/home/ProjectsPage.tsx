import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FilePlus2, Trash2 } from "lucide-react";
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
