using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Runtime.InteropServices;

public class RunDLL : MonoBehaviour
{

    [DllImport("kinect_dll")]
    private extern static int Kinect_viewer();

    // Start is called before the first frame update
    void Start()
    {
        Debug.Log(Kinect_viewer());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
