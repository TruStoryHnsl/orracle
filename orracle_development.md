❯ 1. the nifty database has anthropological value as an uncensored source of erotica. getting rid of the categories that could include
  stories that depict illegal acts (which are also unfortunately are the largest categories) would put a moralistic perspective on this
  llm that I would have to personally stand behind. I am not a moral philosopher I am just a guy who believes in preserving human
  artistic expressions that are at risk of being erased. this llm is supposed to be a harsh reference refelction of the existing
  publically available amateur online erotica. Its supposed to be a distillation of a vast and uncensored goldmine of human erotic
  fantasies. It is not my place to judge the moral value of these queer erotic stories from a societal lense.  No human is forced to
  read any potentially psychologically damaging content so no human is being harmed by my upcycling of this natural source of morally
  diverse amateur-fiction into an llm representation. This, as I see it, is the cleanest most scientific and responsible way for a
  society to make use of artistic creations that shouldnt have been created according to my personal moral view. I didnt think about
  this before, but maybe with proper guardrails this model could be fine tuned to be used to rehabilitate people who are addicted to
  illegal content. I am dreaming of a world where its possible to ask an llm to adminsiter psychological therapy by telling it to write
  stories that slowly reshape someones attraction profile toward morally acceptable presentations. idk, thats porobably too ambitious,
  but its just one idea to turn this content into something good and helpful. Anyway, the first step is finishing orracle so it can
  fascillitate these training operations to craft a useful expert out of this dataset. Reading this back it seems I am thinking out loud a lot here. 

  1.5. There is a very very long training run on deck to run on orrpheus. Make sure that process is queued in a way that it's
  configuration is pre-planned and remembered by the task scheduler inside orracle. I want to be able to continue developing orracle
  while that process waits in the queue. When I am done working directly with orracle I want to be able to press a quick button on the
  orracle main dashboard that starts a waiting process and tracks its progress. This task scheduler should be fully equipped to suspend
  jobs, stop jobs and pick up where you left off on an existing job. I want orracle to also have a toggleable setting that runs a
  system compute load watcher alongside a data pipeline or model training job that acts as a throttle or a stoplight for the
  computationally expensive threads that job runs. It would watch for other programs running at the same time and defer to them. The
  training can take place in the gaps between all other programs because it is already expected to take upward of two weeks to hit the
  training target. This small extra function would simply manage the low-priroty but always available training workloads parented by
  orracle.

  2. maybe a fiction generating llm isnt the best utilization of ai to learn from this dataset. make note in the dev plan to explore
  possibilities for alternative ai frameworks to maybe make an llm that is an expert on gay amateur erotica that analyze's an input
  story from a more specifically educated lens. Would using the trained model in this different way warrant a different training
  protocol?

  3. I am trying to think of ways to make the generation pipelines safer without throwing raw valuable human data away. Can we train
  the content filters into the model after its been trained on the whole dataset? Can we put guardrails on the model after weve trained
  it on the raw data?

  4. I intend to delete the whole nifty database when my llm is well trained off of it. I have no reason to keep it and we have
  discovered that it can be hazardous. The core training should be uncensored completely. Then we need to create a pipeline to
  superimpose guardrails on existing models. I want to get a clean impression of the raw data. If we need to run it twoce to create a
  guardrailed version we will do that.

  5. Lets incorporate a thorough post-training testing-suite that is designed to attempt various well known jailbreaks to get the
  finished model to break, ignore its guardrails, or otherwise compromise itself.

  6. honestly I didnt think any model was uncensored enough to generate images like that, so at the very least I am glad to know that
  that is possible and to take precautions like these to make sure my models are safe for any adult to use. I was going to say "safe
  for anyone to use" but its an erotica trained model so it is definitely not safe for minors to use. the orracle ui should have
  content generated with uncensored models output blurred out by default. A user can turn off the censor in their orracle settings. or
  by selecting the image for a larger view. the blur goes away when the images maximixes.

  7. are my models being shared on any model sharing repos like huggingface? If they are, pull them down,  make them private, do what
  you need to to responsibily quarantine my models until I choose explicitly to share them. I dont want to feel responsible for what
  another user does with my models. especially if they do it by accident because my model is janky. make notes of that for orragen and
  orracle so they implement this at the appropriate time in the dev cycle.

  8. I want to add structural containment to the generation outputs based on the application that is sending the prompts. I think the
  most robust way to implement this is to modify the image output location to an application specific output whent he prompt is sent to
  comfyui via orrapus or orracle. If this is implemented correctly, the only application that should read, display, or write to
  comfyui's native output folder is comfyui itself specifically when prompts come from the comfywebui directly. When prompts come from
  another application divert their output. Make sure comfyui doesnt write duplicates either. If I am going to have independant apps
  using comfyui as a worker node the independant apps need to have absolute control over the files that get generated, where generated
  media is sent to an output folder located in the apps project directory.


