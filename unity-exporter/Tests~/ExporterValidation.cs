#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEditor;
using RenderFormerStudy;
public static class ExporterValidation
{
    [Serializable] public class SceneData
    {
        public CameraData[] cameras;
    }
    [Serializable] public class CameraData
    {
        public float[] position,look_at,up;
        public float fov;
    }
    static Vector3 Parse(string line)
    {
        var a=line.Split(' ');
        return new Vector3(float.Parse(a[1],System.Globalization.CultureInfo.InvariantCulture),float.Parse(a[2],System.Globalization.CultureInfo.InvariantCulture),float.Parse(a[3],System.Globalization.CultureInfo.InvariantCulture));
    }
    static void CheckFace(string file)
    {
        var lines=File.ReadAllLines(file);
        var vs=lines.Where(x=>x.StartsWith("v ")).Select(Parse).ToArray();
        var ns=lines.Where(x=>x.StartsWith("vn ")).Select(Parse).ToArray();
        var ids=lines.First(x=>x.StartsWith("f ")).Split(' ').Skip(1).Select(x=>int.Parse(x.Split('/')[0])-1).ToArray();
        Assert(Vector3.Dot(Vector3.Cross(vs[ids[1]]-vs[ids[0]],vs[ids[2]]-vs[ids[0]]).normalized,ns[ids[0]])>.999f,"Face normal orientation");
    }
    static void Assert(bool x,string s)
    {
        if(!x)throw new Exception(s);
    }
    public static void Run()
    {
        try
        {
            string dir="F:/RenderFormer-work/unity-export-test-results";
            Directory.CreateDirectory(dir);
            var mesh=new Mesh();
            mesh.vertices=new[]
            {
                Vector3.zero,Vector3.right,Vector3.up
            }
            ;
            mesh.normals=new[]
            {
                Vector3.forward,Vector3.forward,Vector3.forward
            }
            ;
            mesh.triangles=new[]
            {
                0,1,2
            }
            ;
            StaticSceneExporter.WriteMesh(mesh,0,Matrix4x4.identity,dir+"/identity.obj");
            Assert(File.ReadAllText(dir+"/identity.obj").Contains("f 1//1 3//3 2//2"),"Axis parity");
            StaticSceneExporter.WriteMesh(mesh,0,Matrix4x4.Scale(new Vector3(-2,3,4)),dir+"/negative.obj");
            Assert(File.ReadAllText(dir+"/negative.obj").Contains("f 1//1 2//2 3//3"),"Negative scale parity");
            CheckFace(dir+"/identity.obj");
            CheckFace(dir+"/negative.obj");
            mesh.normals=Enumerable.Repeat(new Vector3(1,1,1).normalized,3).ToArray();
            StaticSceneExporter.WriteMesh(mesh,0,Matrix4x4.Scale(new Vector3(2,3,4)),dir+"/normal.obj");
            var line=File.ReadLines(dir+"/normal.obj").First(l=>l.StartsWith("vn ")).Split(' ');
            var n=new Vector3(float.Parse(line[1],System.Globalization.CultureInfo.InvariantCulture),float.Parse(line[2],System.Globalization.CultureInfo.InvariantCulture),float.Parse(line[3],System.Globalization.CultureInfo.InvariantCulture));
            Assert(Vector3.Distance(n,new Vector3(0.5f,0.25f,1f/3f).normalized)<1e-5f,"Inverse transpose normal");
            var go=GameObject.CreatePrimitive(PrimitiveType.Cube);
            var mat=new Material(Shader.Find("Universal Render Pipeline/Lit"));
            foreach(var p in new[]
            {
                "_BaseMap","_BumpMap","_MetallicGlossMap","_SpecGlossMap","_OcclusionMap","_EmissionMap","_ParallaxMap","_DetailAlbedoMap","_DetailNormalMap"
            }
            )if(mat.HasProperty(p))mat.SetTexture(p,null);
            mat.SetFloat("_Metallic",0);
            mat.SetFloat("_Smoothness",0.5f);
            go.GetComponent<MeshRenderer>().sharedMaterial=mat;
            var c=new GameObject("Camera").AddComponent<Camera>();
            c.aspect=1;
            c.transform.position=new Vector3(0,0,-3);
            c.transform.LookAt(Vector3.zero);
            string output=dir+"/cube-"+DateTime.UtcNow.ToString("yyyyMMddHHmmss");
            StaticSceneExporter.Export(new[]
            {
                go
            }
            ,c,output);
            Assert(File.Exists(output+"/scene.json"),"JSON export");
            var scene=JsonUtility.FromJson<SceneData>(File.ReadAllText(output+"/scene.json"));
            var cam=scene.cameras[0];
            Assert(Mathf.Abs(cam.position[1]+3)<1e-5f&&Mathf.Abs(cam.look_at[1]+2)<1e-5f&&Mathf.Abs(cam.up[2]-1)<1e-5f&&Mathf.Abs(cam.fov-c.fieldOfView)<1e-5f,"Camera JSON pose and FOV");
            mat.SetTexture("_BaseMap",Texture2D.whiteTexture);
            bool rejected=false;
            try
            {
                StaticSceneExporter.Export(new[]
                {
                    go
                }
                ,c,dir+"/invalid");
            }
            catch(Exception)
            {
                rejected=true;
            }
            Assert(rejected,"Texture rejection");
            File.WriteAllText(dir+"/validation.json","{\"passed\":true,\"checks\":[\"axis parity\",\"negative scale parity\",\"inverse transpose normals\",\"cube export\",\"texture rejection\",\"OBJ cross-product orientation identity and negative scale\",\"camera JSON position target up FOV\"],\"cube_path\":\""+output+"\"}");
            File.WriteAllText(dir+"/validation-provenance.json", "{\"utc\":\""+DateTime.UtcNow.ToString("O")+"\",\"source_sha256\":\""+BitConverter.ToString(System.Security.Cryptography.SHA256.Create().ComputeHash(File.ReadAllBytes("Assets/Editor/ExporterValidation.cs"))).Replace("-", "")+"\"}");
                EditorApplication.Exit(0);
        }
        catch(Exception e)
        {
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }
}
#endif
