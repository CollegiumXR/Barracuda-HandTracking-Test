﻿/* LICENSE
Copyright (c) 2019 Adrian Babilinski
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/* This class allows you to resize an image on the GPU. 
Resizing an image from 1024px to 8196px 100 times took this method: 00:00:40.8884790
Resizing an image from 1024px to 8196px 100 times with Unity.Texture2D.Resize() took: 01:08:08.55
*/
using System.Collections.Generic;
using System.Threading;
using System.IO;
using UnityEngine;
public class ResizeTool
{
    public static void Resize(Texture2D texture2D, int targetX, int targetY, bool mipmap = true, FilterMode filter = FilterMode.Bilinear)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetX, targetY, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);


        RenderTexture.active = rt;
        Graphics.Blit(texture2D, rt);
        texture2D.Resize(targetX, targetY, texture2D.format, mipmap);
        texture2D.filterMode = filter;

        try
        {
            texture2D.ReadPixels(new Rect(0.0f, 0.0f, targetX, targetY), 0, 0);
            texture2D.Apply();
        }
        catch
        {
            Debug.LogError("Read/Write is not enabled on texture " + texture2D.name);
        }


        RenderTexture.ReleaseTemporary(rt);
    }
    /// <summary>
    /// Gets generic texture  
    /// </summary>
    /// <param name="in_Texture"></param>
    /// <param name="out_Texture"></param>
    /// <param name="targetX"></param>
    /// <param name="targetY"></param>
    /// <param name="mipmap"></param>
    /// <param name="filter"></param>
    public static void Resize(Texture in_Texture, Texture2D out_Texture, int targetX, int targetY, bool mipmap = true, FilterMode filter = FilterMode.Bilinear)
    {

        //FGraphics.CopyTexture(in_Texture,out_Texture);
        RenderTexture rt = RenderTexture.GetTemporary(targetX, targetY, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);


        RenderTexture.active = rt;
        Graphics.Blit(in_Texture, rt); // Copies source texture into destination render texture with a shader (on the gpu) 
        out_Texture.Resize(targetX, targetY, out_Texture.format, mipmap); //  resize the texture to the target values (this sets the pixel data as undefined)
        out_Texture.filterMode = filter;

        try
        {
            out_Texture.ReadPixels(new Rect(0.0f, 0.0f, targetX, targetY), 0, 0); // reads the pixel values from the temporary RenderTexture onto the resized texture
            out_Texture.Apply(); // actually upload the changed pixels to the graphics card 
        }
        catch
        {
            Debug.LogError("Read/Write is not enabled on texture " + out_Texture.name);
        }


        RenderTexture.ReleaseTemporary(rt);
    }


}
