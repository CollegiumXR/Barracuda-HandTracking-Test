using System.Linq;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class DepthSensor : MonoBehaviour
{
    [SerializeField] private NNModel _monoDepthONNX;
    [SerializeField] private RawImage _sourceImageView;
    [SerializeField] private RawImage _destinationImageView;
    private Model m_RuntimeModel;
    private IWorker worker;
    private WebCamTexture _webCamTexture;
    private RenderTexture outputRenderTexture;
    private int channelCount = 3;
    private RenderTexture frame;
    private Texture2D inputTexture;
    private Texture2D depthTexture;
    private Rect region;
    private int modelwidth = 224;
    private int modelheight = 224;
    private ResizeTool _resizeTool;

    private Vector3[] vertices;
    private int[] triangles;
    private Mesh mesh;
    private Color[] colors;

    public RenderTexture tensorTarget;
    public Renderer depthMaterialRenderer;


    private void Start()
    {
        InitWebCamFeed();
        InitBarracuda();
        InitResizerAndTextures();
    }

    private void InitBarracuda()
    {
        m_RuntimeModel = ModelLoader.Load(_monoDepthONNX);
        worker = WorkerFactory.CreateComputeWorker(m_RuntimeModel);
    }

    private void InitWebCamFeed()
    {
        _webCamTexture = new WebCamTexture(620, 480, 30);
        _sourceImageView.texture = _webCamTexture;
        _webCamTexture.Play();
    }

    private void InitPointCloudMesh()
    {
        vertices = new Vector3[modelwidth * modelheight];
        colors = new Color[modelwidth * modelheight];
    }

    private void InitResizerAndTextures()
    {
        inputTexture = new Texture2D(modelwidth, modelheight, TextureFormat.RGB24, false);
        depthTexture = new Texture2D(modelwidth, modelheight, TextureFormat.RGB24, false);
        region = new Rect(0, 0, modelwidth, modelheight);
    }


    private void Update()
    {

        ResizeWebCamFeedToInputTexture();

        var tensor = new Tensor(inputTexture);
        // inference
        var output = worker.Execute(tensor).PeekOutput();
        float[] depth = output.AsFloats();
        Debug.Log("Min : " + depth.Min() + "Max : " + depth.Max());
        // PrepareDepthTextureFromFloats(depth);
        feedTheMinMaxToMaterial(depth);
        output.ToRenderTexture(tensorTarget);
        //_destinationImageView.texture = tensorTarget;
        tensor.Dispose();

    }

    private void ResizeWebCamFeedToInputTexture()
    {
        //Resize the webcam texture into the input shape dimensions
        ResizeTool.Resize(_webCamTexture,inputTexture,modelheight,modelwidth);
    }
    private void feedTheMinMaxToMaterial(float[] depth){
        var min = depth.Min();
        var max = depth.Max();

        _destinationImageView.material.SetFloat("_min", min);
        _destinationImageView.material.SetFloat("_max", max);
    }
/// <summary>
/// old Method
/// </summary>
/// <param name="depth"></param> 
/*
    private void PrepareDepthTextureFromFloats(float[] depth)
    {
        var min = depth.Min();
        var max = depth.Max();
        foreach (var pix in depth.Select((v, i) => new { v, i }))
        {
            var x = pix.i % modelwidth;
            var y = pix.i / modelwidth;
            var invY = modelheight - y - 1;

            // normalize depth value
            var val = (pix.v - min) / (max - min);
            depthTexture.SetPixel(x, y, new Color(val, 0.59f * val, 0.11f * val));
            var worldPos = new Vector3(x / (modelwidth / 0.9f), y / (modelheight / 0.9f), val);
            vertices[y * modelwidth + x] = worldPos;
            //colors[y * modelwidth + x] = inputTexture.GetPixel(x, invY);
        }
        depthTexture.Apply();
    }
*/

}