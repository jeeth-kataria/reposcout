import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const results = path.join(root, "results");
const output = path.join(root, "web/public");
await mkdir(output, {recursive: true});
for (const asset of ["index.html", "app.js", "styles.css"]) {
  await cp(path.join(root, "web", asset), path.join(output, asset));
}
const readJson = async (file) => JSON.parse(await readFile(file, "utf8"));
const readRows = async (file) => (await readFile(file, "utf8")).trim().split("\n")
  .filter(Boolean).map(line => JSON.parse(line));
const runs = [];
for (const entry of await readdir(results, {withFileTypes: true})) {
  if (!entry.isDirectory()) continue;
  const dir = path.join(results, entry.name);
  const config = await readJson(path.join(dir, "config.json"));
  const files = await readdir(dir);
  if (!files.includes("metrics.json")) continue;
  const metrics = await readJson(path.join(dir, "metrics.json"));
  if (!metrics.instances || metrics.instances !== metrics.target_instances) continue;
  const predictions = await readRows(path.join(dir, "predictions.jsonl"));
  const gold = new Map((await readRows(path.join(dir, "gold.jsonl"))).map(i => [i.instance_id, i]));
  if (predictions.length !== metrics.instances) throw Error("Prediction count differs: " + entry.name);
  const cases = predictions.map(pred => {
    const issue = gold.get(pred.instance_id);
    if (!issue) return {id: pred.instance_id, status: pred.stats.status, unavailable: true};
    return {id: pred.instance_id, repo: issue.repo, commit: issue.base_commit, issue: issue.problem_statement,
      goldFiles: issue.gold_files, goldFunctions: issue.gold_functions, files: pred.files,
      functions: pred.functions, candidates: pred.candidates, stats: pred.stats};
  });
  const filename = encodeURIComponent(entry.name) + ".json";
  await writeFile(path.join(output, filename), JSON.stringify(cases));
  runs.push({name: entry.name, localizer: config.localizer, split: config.split, model: config.model,
    count: metrics.instances, metrics, file: filename});
}
if (!runs.length) throw Error("A completed recorded run is required.");
runs.sort((a, b) => Number(b.split === "dev") - Number(a.split === "dev") ||
  b.count - a.count || a.name.localeCompare(b.name));
await writeFile(path.join(output, "runs.json"), JSON.stringify(runs));
console.log("Exported", runs.length, "recorded runs");
