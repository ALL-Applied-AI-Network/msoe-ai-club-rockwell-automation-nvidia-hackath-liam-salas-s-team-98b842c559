# Rockwell Automation + NVIDIA Innovation Lab Fall 2026
### Authors: MSOE AI Club

### A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute. Teams pick one of three challenges from a real factory testbed (vial inspection, fluid mixing, or robotic grasping and sorting) and build solutions with Rockwell and NVIDIA mentors and NVIDIA compute.

# Physical AI for the Factory Floor
[Rockwell Automation](#) , [NVIDIA](#) , the [MSOE AI Club](#) , and UWM's [Connected Systems Institute (CSI)](#) are running a four-week physical AI hackathon. Teams will work with data and equipment from CSI's manufacturing testbed, then show a working solution at the finals.

You do not need prior manufacturing or AI experience. You do need a team that is willing to test ideas, measure results, and explain what it learned.

**Join the Innovation Lab**
Each student creates an account and joins or creates a team.
* [Sign up and manage your team](#)
* [Submit your team plan](#) by 11:59 PM CT on Wednesday, October 14.
* [Open the event workspace](#)

*Use the same email address associated with your university if possible.* After you join a team, connect GitHub from your profile so the dashboard can send your team repository invitation.

### How teams work on the ALL AI Network
Teams, repositories, and submissions for this lab are managed through the [ALL AI Network](#) event workspace.
- *MSOE and UWM students:* your teams have already been created. Create an account with your university email from the sign-up link above, and you will be placed on your team automatically.
- *Not on a team yet?* Create a team in the workspace and invite your teammates, or join an open team.
- *Connect GitHub:* once you connect GitHub from your profile, you are automatically invited to your team's repository. No one has to add you by hand.
- *One place for every project:* all team repositories live in one place, so Rockwell Automation and NVIDIA can view your work as it progresses, and your final submission links straight to it.

**What is the factory testbed?**
CSI operates a high-mix, low-volume demonstration plant built with Rockwell Automation hardware. The production line creates vials filled with mixed-color fluid from custom recipes. Several robots move the vials through filling, inspection, and final placement.

CSI also has a separate FANUC robot setup that moves and sorts blocks and other objects. That robot is *not connected to the vial production line*. It is the physical system and digital-twin setup used for Dataset 3 discussed below.

![Time-lapse overview of the CSI manufacturing testbed](#)

**About our sponsors**
### Rockwell Automation
[Rockwell Automation](#) builds the industrial automation and control systems that run production lines across manufacturing, from the controllers on the plant floor to the FactoryTalk software that connects production data and workflows. Its Emulate3D tools let engineers model and test a manufacturing system as a digital twin before they change physical equipment.

Rockwell's customers are under pressure to produce more product variants, in smaller batches, with fewer experienced people on the floor. A line that changes recipes or products often cannot wait weeks for a new inspection program or robot routine. For Rockwell, AI is valuable when it makes those changeovers faster and more reliable, and when operators can understand and trust what it is doing. Results that hold up on real equipment, with clear limits, matter more than results that only work in a clean demo.

### NVIDIA
[NVIDIA](#) provides the accelerated computing, AI models, simulation frameworks, and edge platforms used to build Physical AI systems—systems that can perceive, reason, plan, and act in the physical world. Across computer vision, robotics, and industrial automation, teams can train models, simulate and validate their behavior with NVIDIA Omniverse and Isaac Sim, generate synthetic data where real-world data is limited, and deploy them for real-time inference and control.

NVIDIA sees Physical AI as the next major wave of computing. In manufacturing, that shift becomes tangible: teams can develop and train models, test them against a physically based digital twin of a production line, and then deploy validated systems into the physical production environment. This approach helps address some of manufacturing’s hardest challenges, including limited labeled data, frequent product and process changes, and the need for systems that operate safely and reliably alongside people and machines.

### Why they are working together
Rockwell brings deep knowledge of industrial operations and the systems manufacturers already run. NVIDIA brings the compute and simulation tools to move AI from experimentation into production. Together, they are bringing NVIDIA's simulation and AI technology into Rockwell's digital twin and design tools, so manufacturers can design, test, and improve automated systems before touching the physical line.

### Why this matters
Most manufacturers have not yet adopted AI on the factory floor. Only about 3% of manufacturers used AI for automated inspection in the 2020 Annual Business Survey. A NIST survey estimated that defective products cost US discrete manufacturers \$32–58 billion each year. The gap between what AI can do and what is actually running in plants is large, and much of it comes down to reliability, setup time, and trust.

This lab asks teams to close a small piece of that gap: pick one real process on CSI's testbed, make it measurably better, and show the result working. The [MSOE AI Club](#) and UWM-CSI are bringing these tools and this testbed to student teams through this Innovation Lab.

**The three datasets**
We recommend that teams focus on building one strong solution around a single idea or dataset rather than spreading effort across several half-finished ideas. This is a recommendation, not a requirement, but it is the approach that has worked best for teams in past Innovation Labs.

All of the data and digital twins are in the [Innovation Lab data folder](#). The real data from the line is there to look at, not to train on: teams generate their training data with the digital twins, as a high-mix, low-volume line would have to. Each dataset page explains why.

### Dataset 1: Final Vial Inspection
Find missing caps, crooked caps, damaged vials, and unexpected empty positions after the robot fills a fixture.
[See more details](#)

### Dataset 2: Fluid Mixing Station 3
Use synchronized camera views to detect fill and color problems caused by recipes and fluid carryover.
[See more details](#)

### Dataset 3: Grasping and Sorting
Use a digital twin and multiple cameras to make a standalone FANUC arm grasp and sort objects more reliably. This robot is separate from the vial production line.
[See more details](#)

### Related guides
These four guides cover what every team needs beyond the datasets. Read them before you submit your team plan.
- [Compute: Brev, DGX Spark, and build.nvidia.com](#): how to get GPU power for your project. It covers Brev cloud GPUs and ready-made launchables (including Isaac Sim), the DGX Spark systems at UWM and Dell GB10 systems at MSOE, free model APIs on build.nvidia.com, and the workstation at CSI. Brev credits are spent whenever an instance is running, so read this before you start one.
- [Collect and test your own data](#): what you can and cannot do at the CSI testbed. It explains how to request a visit or extra data, how to collect your own data safely, how to use digital twins for synthetic data, and how to schedule a supervised test of your finished system on the real equipment.
- [Coding agents](#): start free with Gemini in the Antigravity CLI, learn how to get more student usage, and request Claude or Codex plans in your team plan.
- [Tools and resources](#): a table matching each physical AI task to its tools and hardware, a step-by-step tutorial for synthetic data generation with Isaac Sim on Brev, AI coding tools, in-context learning with vision-language models, NVIDIA Cosmos, and other tools worth knowing. Start with a simple baseline, then use this page when you hit a specific question.

**Schedule**

| *Date* | *Milestone* | *Details* |
| :--- | :--- | :--- |
| Thursday, October 8 | Kickoff | Rockwell Automation HQ, 1201 S 2nd St, Milwaukee, 4:00–6:00 PM. Transportation is provided from both campuses. |
| Wednesday, October 14 | Team plan due | Submit by 11:59 PM CT through the [team plan submission form](#). |
| Sunday, November 1 | Final submission due | About a five-minute video, a one-page summary, your repository, and references. |
| Thursday, November 5 | Finals | MSOE Diercks Hall, 5:00–8:00 PM. More details to come. |

**First week and team plan**
### First-week checklist
- [ ] Create an All AI Network account with the email address associated with your AI Club points.
- [ ] Create or join a team and settle on a team name.
- [ ] Connect your GitHub account and accept the team repository invitation.
- [ ] Choose dataset(s).
- [ ] Download the data and confirm that everyone who needs it can open it.
- [ ] Review the [compute guide](#) and request what your team expects to use inside the [team plan](#).
- [ ] [Join the hackathon Slack workspace](#), open `#hackathon-announcements`, and find the channel for your dataset.
- [ ] Submit the [team plan](#) by Wednesday, October 14.

### Team plan
[Submit your team plan here](#) by 11:59 PM CT on Wednesday, October 14.

The plan is a short resource request, not a contract or graded paper. Anyone on the team can submit it, and any teammate can update the same team submission before the deadline. Organizers and sponsor mentors will use it to line up compute, data, and technical support.

The form asks for:
- your team name and members;
- the dataset or combination of datasets you plan to use;
- a short description of the problem you plan to solve;
- data you need beyond what is already provided, and why;
- the compute you expect to use: Brev (including how many credits your team is requesting), a UWM DGX Spark, an MSOE Dell GB10 on Rosie, the CSI workstation, or build.nvidia.com;
- [AI-agent credits or tools](#) you need;
- the type of mentor support you want;
- questions for the organizers.

Plans may change once teams examine the data. Keep the first submission short and specific enough for sponsors to understand what help would be useful.

**Final submission**
[Submit your final project here](#) by 11:59 PM CT on Sunday, November 1.

Each team submits:
- a video of about five minutes;
- a one-page project summary;
- a link to the team repository;
- references and credits for data, models, code, and tools.

### What judges expect
Your video, summary, and finals presentation should:
- *Demonstrate the solution operating on real hardware*, either the UWM CSI production line or the FANUC robotic arm. See the [data collection guide](#) to schedule a supervised test.
- *Identify the specific problem being solved*, such as a defect type, a failure mode, or a slow manual step, and why it matters on this line.
- *Explain how the solution would impact other manufacturers.* Think about the business case: what the problem costs today (scrap, rework, downtime, or labor), what your solution would save or improve, and what it would take for a plant to adopt it (hardware, setup time, changeovers, and operator trust).
- *Show measured results* against a baseline, and be clear about where the solution still fails.

The Nov. 5 finals include live presentations and prizes.

**Prizes**

TOTAL PRIZE POOL
***\$5,000 in cash***

**RTX 5080**
NVIDIA GeForce graphics card

**RTX 5070**
NVIDIA GeForce graphics card

**Sponsor swag**
from Rockwell Automation and NVIDIA

*Details to come* on how the cash is split between first, second, and third place, and on the additional award categories and prizes.

**Judging**
Projects will be reviewed by judges representing *Rockwell Automation, NVIDIA, MSOE, and UWM*.

Judges will look for the expectations listed under *Final submission*: a solution operating on the CSI line or robotic arm, a specific problem, a clear business case for other manufacturers, and measured results.

The detailed judging criteria, category weights, and award categories are still being finalized. More information will be shared when it is available.

**Getting help**
This year, teams will not be assigned one exact mentor or given a long directory of sponsor contacts. [Join the hackathon Slack workspace](#), then ping the people below in the channel that matches the question. This keeps answers visible to other teams. Email is available if you do not get a response in Slack or would rather follow up directly.

### Recommended Slack channels to create
- `#hackathon-announcements` — organizer updates, deadlines, and schedule changes.
- `#hackathon-general` — general event questions and help finding the right channel.
- `#dataset-1-vial-inspection` — questions and findings about final vial inspection.
- `#dataset-2-fluid-mixing` — questions and findings about Station 3 and fluid recipes.
- `#dataset-3-grasping-sorting` — questions about the standalone FANUC robot and digital twin.
- `#data-and-testbed-requests` — additional data, CSI visits, on-site collection, and live testing.
- `#compute-questions` — Brev credits, instances, DGX Spark and GB10 access, and compute troubleshooting.

### Who to ping
- *More data, CSI visits, or physical testing:* Ping *Shamar Webster* in `#data-and-testbed-requests`. This includes requesting more production-line data, visiting CSI to collect your own data, and arranging a supervised test of your tool. Email fallback: [webste63@uwm.edu](mailto:webste63@uwm.edu).
- *Questions about the line or datasets:* Ping *Brett Storoe, Dr. Derek Riley, or Tanner Cellio* in the relevant dataset channel. These are the first contacts for non-critical questions about the equipment, dataset context, or problem your team is trying to understand. Email fallbacks: [storoeb@msoe.edu](mailto:storoeb@msoe.edu) and [driley@nvidia.com](mailto:driley@nvidia.com).
- *Questions about the FANUC robot (Dataset 3):* Email *Zhen Zeng* at [zhenzeng@uwm.edu](mailto:zhenzeng@uwm.edu), or ask in `#dataset-3-grasping-sorting`.
- *Brev or compute:* Ping *Dan Schneider* in `#compute-questions`, or email [danschneider@nvidia.com](mailto:danschneider@nvidia.com).
- *Website, signup, team, or Slack access:* Ping *Brett Storoe* in `#hackathon-general`, or email [storoeb@msoe.edu](mailto:storoeb@msoe.edu).

For a critical-path issue that is blocking the team, say that clearly in the first line of the Slack message and include what you tried, the error, and when you need the issue resolved.