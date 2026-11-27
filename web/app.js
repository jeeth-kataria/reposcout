const element = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const link = (label, href) => { const a = element("a", label); a.href = href; return a; };
const read = async file => {
  const response = await fetch(file);
  if (!response.ok) throw Error("Could not load " + file);
  return response.json();
};
const select = document.querySelector("#runs");
const filter = document.querySelector("#filter");
const pane = document.querySelector("#case");
const list = document.querySelector("#issues");
let runs = [], issues = [], selected, tab = "Files", request = 0;
const metric = (value, label) => {
  const card = element("div", undefined, "metric");
  card.append(element("strong", value), element("span", label)); return card;
};
function showList() {
  list.replaceChildren();
  const matches = issues.filter(i => i.id.toLowerCase().includes(filter.value.toLowerCase()));
  if (!matches.length) list.append(element("li", "No issues match."));
  for (const issue of matches) {
    const button = element("button", issue.id.replace("__", "/"));
    button.classList.toggle("active", selected === issue);
    button.setAttribute("aria-current", String(selected === issue));
    button.onclick = () => {selected = issue; showList(); showIssue();};
    const li = element("li"); li.append(button); list.append(li);
  }
}
function showIssue() {
  pane.replaceChildren();
  const i = selected;
  if (!i) return;
  if (i.unavailable) {pane.append(element("h2", i.id), element("p", "Source metadata unavailable for this run.")); return;}
  const repoURL = "https://github.com/" + i.repo.split("/").map(encodeURIComponent).join("/");
  const heading = i.issue.trim().split("\n").find(line => line.trim()) || i.id;
  const meta = element("div", undefined, "eyebrow");
  meta.append(link(i.repo, repoURL), " / ", link(i.commit.slice(0, 8), repoURL + "/commit/" + i.commit));
  const full = element("details", undefined, "details");
  full.append(element("summary", "Issue statement"), element("p", i.issue, "issuebody"));
  pane.append(meta, element("h2", heading), full, element("p",
    i.stats.status + " · " + i.stats.llm_calls + " LLM calls · " + i.stats.latency_s.toFixed(3) + " s localization", "meta"));
  if (i.stats.error) pane.append(element("p", i.stats.error));
  const tabs = element("div", undefined, "tabs");
  for (const name of ["Files", "Functions"]) {
    const b = element("button", name + " (" + i[name.toLowerCase()].length + ")");
    b.setAttribute("aria-pressed", String(tab === name)); b.onclick = () => {tab = name; showIssue();};
    tabs.append(b);
  }
  pane.append(tabs);
  const files = tab === "Files";
  const ranked = files ? i.files : i.functions;
  for (const [idx, item] of ranked.entries()) {
    const row = element("details", undefined, "rank");
    const title = element("summary", String(idx + 1).padStart(2,"0") + "  " + item);
    if ((files ? i.goldFiles : i.goldFunctions).includes(item)) title.append(element("span","Gold match","gold"));
    row.append(title);
    const candidate = i.candidates.find(c => files ? c.path === item : c.symbol_id === item);
    if (candidate) {
      row.append(element("p", candidate.rationale));
      const cites = element("ul");
      for (const c of candidate.citations) {
        const li = element("li");
        const source = repoURL + "/blob/" + i.commit + "/" + c.path.split("/").map(encodeURIComponent).join("/") +
          "#L" + c.start_line + "-L" + c.end_line;
        li.append(link(c.path + " · L" + c.start_line + "–" + c.end_line, source)); cites.append(li);
      }
      row.append(cites);
    }
    row.open = idx === 0;
    pane.append(row);
  }
}
async function showRun() {
  const id = ++request, run = runs[select.selectedIndex];
  const metrics = document.querySelector("#metrics");
  document.querySelector("#run-model").textContent = run.model || run.localizer;
  const pct = n => n === null ? "—" : (100*n).toFixed(1).replace(/\.0$/,"") + "%";
  metrics.replaceChildren(metric(run.count, run.split + " instances"), metric(pct(run.metrics["file_hit@5"]),"File hit @ 5"),
    metric(pct(run.metrics["func_recall@5"]),"Function recall @ 5"), metric(run.metrics.mean_tokens,"Mean tokens"));
  issues=[]; selected=undefined; showList(); pane.textContent="Loading issues…";
  const data = await read("/" + run.file);
  if (request !== id) return;
  issues = data; selected = issues[0]; filter.value=""; showList(); showIssue();
}
try {
  runs = await read("/runs.json");
  for (const run of runs) select.add(new Option(run.name, run.name));
  select.onchange = () => showRun().catch(e => {pane.textContent=e.message;});
  filter.oninput = showList;
  await showRun();
} catch (error) { pane.textContent = error.message; }
