using System.Collections;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;

public class GridArea : MonoBehaviour
{
    private EnvironmentParameters environmentParameters;

    private List<GameObject> gameObjects;

    /// <summary>
    /// The camera  for visual representation of the grid area
    /// </summary>
    private Camera agentCamera;

    [Tooltip("The agent that will be trained")]
    public GameObject agent;
    public GameObject target;
    public GameObject fire;

    private GameObject[] environmentObjectTypes;
    private int[] environmentObjects;

    /// <summary>
    /// Grabs environment parameters and initializes components for use later
    /// </summary>
    void Start()
    {
        environmentParameters = Academy.Instance.EnvironmentParameters;

        // Determine if we need to delete the light source
        bool useLight = Convert.ToBoolean(environmentParameters.GetWithDefault("allow_light_source", 1.0f));
        if (!useLight) {
            Light[] lights = FindObjectsOfType<Light>();
            if (lights.Length == 1) {
                DestroyImmediate(lights[0]);
            }
        }

        agentCamera = transform.Find("AgentCamera").GetComponent<Camera>();
        environmentObjectTypes = new[] { target, fire };
        gameObjects = new List<GameObject>();

        UnityEngine.Random.InitState(System.Environment.TickCount);
    }

    /// <summary>
    /// Creates an environment.
    /// Sets values for targets
    /// </summary>
    private void CreateEnvironment() {
        List<int> objs = new List<int>();
        // Add a number of Target objects
        int numTargets = (int)environmentParameters.GetWithDefault("num_targets", 2);
        for (int i = 0; i < numTargets; i++) {
            objs.Add(0); // 0 index in the environmentObjectTypes[]
        }
        // Add a number of Fire objects
        int fireTargets = (int)environmentParameters.GetWithDefault("num_fires", 4);
        for (int i = 0; i < fireTargets; i++) {
            objs.Add(1); // 1 index in the environmentObjectTypes[]
        }

        environmentObjects = objs.ToArray();
    }

    /// <summary>
    /// Resets the grid area
    /// </summary>
    public void ResetArea() {
        // Destroy old game objects
        foreach(GameObject obj in gameObjects) {
            DestroyImmediate(obj);
        }
        // Create new environment
        CreateEnvironment();
        gameObjects.Clear();

        // Generate a random location on the grid for each fire and target + one agent.
        // The first entry will be for the agent's spawn location.
        HashSet<int> vals = new HashSet<int>();
        vals.Add(UnityEngine.Random.Range(0, 8*8));  // Agent
        while (vals.Count < environmentObjects.Length + 1) {
            vals.Add(UnityEngine.Random.Range(0, 8*8));
        }
        int[] generatedVals = vals.ToArray();

        int xLoc, yLoc;
        // Instantiate the objects at the random locations throughout the grid.
        // The first generate value is for the agent, so just skip over it.
        for (int i = 1; i < generatedVals.Length; i++) {
            xLoc = generatedVals[i] / 8;
            yLoc = generatedVals[i] % 8;
            GameObject obj = Instantiate(environmentObjectTypes[environmentObjects[i - 1]], transform);
            obj.transform.localPosition = new Vector3(xLoc - 3.5f, 0.5f, yLoc - 3.5f);
            gameObjects.Add(obj);
        }
        // Move the agent
        xLoc = generatedVals[0] / 8;
        yLoc = generatedVals[0] % 8;
        agent.transform.localPosition = new Vector3(xLoc - 3.5f, 0.5f, yLoc - 3.5f);
    }
}
