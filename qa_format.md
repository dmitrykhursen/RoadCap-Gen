# Answer formats

## The initial question (tag 2 - language)

Every scene in both train and val starts with

**What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.**

so it is not considered in the distribution, but rather manually inserterd at the start of each scene.

The format is:
What (sometimes with color features etc) and where for all the important objects followed by a list of the IDs corresponding to the *key_object_infos* section of the train

- There is a car to the back of the ego vehicle, a small car to the front of the ego vehicle, and a white commercial vehicle to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,755.8,495.8>, <c2,CAM_FRONT,850.0,475.8>, and <c3,CAM_FRONT,1235.8,516.7>.

## Yes/No questions -- Yes./No. (tag 0 - accuracy)

- Would \<obj\> be in the moving direction of the ego vehicle? -- 20.97%
- Will \<obj\> be in the moving direction of \<obj\>? -- 10.78%
- Will \<obj\> change its motion state based on \<obj\>? -- 8.97%
- Would \<obj> take \<obj> into account -- 1.52%

## Multiple options -- A/B/C/D (tag 0 - accuracy)

These questions are open-ended in train data and we don't have the answers for the test server, so the format is taken from [this example on DriveLM github](https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/test_eval.json)

- Predict the behavior of the ego vehicle. Please select the correct answer from the following options: -- 5.44%
- What is the moving status of object \<obj\>? Please select the correct answer from the following options: -- 5.07%

## Individual questions

### What actions could the ego vehicle take based on \<obj\>? Why take this action and what's the probability? -- 19.29% (tag 1 - GPT)

There are some variations, but always starts with *The action is*, then some reason, then the probability. We can try to always separate it into three sentences or just try to enforce all three parts with strict start and leave the rest to the LLM

- The action is to keep going at the same speed. The reason is to follow the traffic rules, which has a high probability
- The action is to keep stationary, the reason is to follow the traffic rules, high
- The action is to keep going at the same speed, the reason is that there is no safety issue. The probability of (taking) this action is high
- The action is to remain stationary. The reason for this action is to follow the traffic rules, which is considered to be high probability
- The action is to remain stationary because the object has little reference value for the ego vehicle, with a high probability

### What actions taken by the ego vehicle can lead to a collision with \<obj\>? -- 9.04% (tag 1 - GPT)

There are two formats. Either only the action, or the whole sentence

- Accelerating and going straight can lead to a collision with \<obj\>
- Sharp left turn
- No such action will lead to a collision
- No actions taken by the ego vehicle will lead to a collision with \<obj\>

### In this scenario, what are safe actions to take for the ego vehicle? --5.44% (tag 1 - GPT)

Comma separated actions. Only saw the full sentence in cca 2%. We can try to enforce wording, or just let the LLM to decide

- Keep going at the same speed, accelerate, and proceed
- Turn left, decelerate gradually without braking
- Accelerate and go ahead, decelerate gradually without braking, slightly offset to the left, and slightly offset to the right
- Accelerate and go ahead, brake suddenly, brake gently to a stop, decelerate gradually without braking, and back up are safe actions to take for the ego vehicle in this scenario

### In this scenario, what are dangerous actions to take for the ego vehicle? -- 5.22% (tag 1 - GPT)

Basically the same as the previous. Same format, same wording, same percentage on the full sentence

### What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take? -- 2.3% (tag 3 - match)

The structure is: Firstly, notice \<obj>. The object is \<action>, so the ego should \<action>. Secondly, notice \<obj>. The object is \<action>, so the ego should \<action>. Thirdly, notice \<obj>. The object is \<action>, so the ego should \<action>.

Sometimes there is *it* instead of *The object*
Sometimes there is *The object is a traffic sign, so the ego should* instead of the \<action>

- Firstly, notice <c2,CAM_FRONT_RIGHT,400.8,628.3>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed. Secondly, notice <c3,CAM_FRONT_LEFT,113.3,545.8>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed. Thirdly, notice <c1,CAM_FRONT,998.3,545.8>. The object is going ahead, so the ego vehicle should keep going ahead at the same speed
- Firstly, notice <c4,CAM_FRONT,807.3,357.2>. The object is a traffic sign, so the ego vehicle should remain stationary. Secondly, notice <c2,CAM_FRONT,138.3,686.7>. The object is stationary, so the ego vehicle should remain stationary. Thirdly, notice <c3,CAM_FRONT,844.2,501.7>. The object is moving ahead, so the ego vehicle should remain stationary

### What's your comment on this scene? -- 2.22% (tag 1 - GPT)

No real pattern to follow

- There is nothing in particular to pay attention to in this scene; the road is quite empty
- The scene is rich in information, the road conditions are complex, and it deserves special attention
- There are many vehicles on both sides of the road in this scene
- The road is relatively empty

### Based on \<obj> in this scene, what is the most possible action of the ego vehicle? --1.36% (tag 1 - GPT)

Usually just one action. From time to time whole sentence

- To yield to pedestrians is the most possible action of the ego vehicle based on <c6,CAM_FRONT,738.7,709.7> in this scene
- Decelerate gradually without braking.
- Keep going at the same speed.
- Brake gently to a stop.
- None.

### Based on the observation of \<obj>, what actions may \<obj> take? -- 0.56% (tag 1 - GPT)

Same as previous

### What is the priority of the objects that the ego vehicle should consider?(in descending order) -- 0.15% (tag 1 - GPT)

Either coma separated list of \<obj>, the sentence and then the list, or the list and then the sentence

### What is the traffic signal that the ego vehicle should pay attention to? -- 0.04% (tag 1 - GPT)

Any sign, Green/Yellow/Red light, None, croswalk, ...

### Questions not in train or the sample

Only in test, so we don't know the format, but the metric is GPT score

- Are there any other issues worth noting in this scene? -- 0.56%
- What situation in this scene affects driving vehicles? -- 0.18%
- What impact does this situation have on driving vehicles? -- 0.15%
- Are there any other notable issues in this scene? -- 0.11%
- What situation in this scene affects the driving of vehicles? -- 0.08%
- What situation in this scene affects driving? -- 0.07%
- What is the impact of this situation on driving vehicles? -- 0.06%
- What impact does the situation have on driving vehicles? -- 0.05%
- What situation affects driving vehicles in this scene? -- 0.05%
There is around 30 more that would end up in this category with slightly different wording in the question, all with the same tag, all with under 0.03% and almost all with 0.01%
