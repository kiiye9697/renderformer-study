#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using System.Globalization;
using UnityEditor;
using UnityEngine;
namespace RenderFormerStudy
{
    public static class StaticSceneExporter
    {
        static string F(float x)
        {
            if(float.IsNaN(x)||float.IsInfinity(x)) throw new Exception("Non-finite value");
            return x.ToString("R",CultureInfo.InvariantCulture);
        }
        static string Q(string s)
        {
            return "\""+s.Replace("\\","\\\\").Replace("\"","\\\"").Replace("\n"," ").Replace("\r"," ")+"\"";
        }
        public static Vector3 ConvertPoint(Vector3 p)
        {
            return new Vector3(p.x,p.z,p.y);
        }
        static string V(Vector3 p)
        {
            return "["+F(p.x)+","+F(p.y)+","+F(p.z)+"]";
        }
        static Vector3 RGB(Color c)
        {
            return new Vector3(c.r,c.g,c.b);
        }
        static bool White(Color c)
        {
            return Mathf.Abs(c.r-c.g)<1e-5f&&Mathf.Abs(c.r-c.b)<1e-5f;
        }
        static string MaterialJson(Material m)
        {
            if(m==null||m.shader.name!="Universal Render Pipeline/Lit") throw new Exception("Only constant URP/Lit materials supported");
            if(m.GetFloat("_Surface")!=0||m.GetFloat("_AlphaClip")!=0) throw new Exception("Transparent/alpha-clipped material unsupported");
            if(m.HasProperty("_Cull")&&m.GetFloat("_Cull")!=2) throw new Exception("Only back-face culling supported");
            foreach(string p in new[]
            {
                "_BaseMap","_BumpMap","_MetallicGlossMap","_SpecGlossMap","_OcclusionMap","_EmissionMap","_ParallaxMap","_DetailAlbedoMap","_DetailNormalMap"
            }
            )
            if(m.HasProperty(p)&&m.GetTexture(p)!=null) throw new Exception("Texture unsupported: "+p);
            if(m.HasProperty("_ClearCoatMask")&&m.GetFloat("_ClearCoatMask")!=0) throw new Exception("Clearcoat unsupported");
            Color b=m.GetColor("_BaseColor");
            if(b.a<0.999f) throw new Exception("Nonopaque base alpha unsupported");
            float spec=0.04f;
            if(m.GetFloat("_WorkflowMode")==0)
            {
                Color s=m.GetColor("_SpecColor");
                if(!White(s))throw new Exception("Colored specular unsupported");
                spec=s.r;
            }
            else if(m.GetFloat("_Metallic")!=0) throw new Exception("Metallic workflow requires metallic=0; use explicit specular workflow");
            Color e=m.IsKeywordEnabled("_EMISSION")?m.GetColor("_EmissionColor"):Color.black;
            if(e.r<0||e.g<0||e.b<0) throw new Exception("Negative emission unsupported");
            if(!White(e)) throw new Exception("Only white mesh emission supported");
            Vector3 d=RGB(b)*(1-spec);
            float r=1-m.GetFloat("_Smoothness");
            if(r<0.01f||r>1||spec<0||spec>1||d.x<0||d.y<0||d.z<0||Mathf.Max(d.x,Mathf.Max(d.y,d.z))+spec>1.00001f)throw new Exception("Material outside supported parameter range");
            return "{\"diffuse\":"+V(d)+",\"specular\":"+V(Vector3.one*spec)+",\"roughness\":"+F(r)+",\"emissive\":"+V(RGB(e))+",\"smooth_shading\":true}";
        }
        public static void WriteMesh(Mesh mesh,int sub,Matrix4x4 world,string path)
        {
            if(mesh.GetTopology(sub)!=MeshTopology.Triangles)throw new Exception("Nontriangle topology");
            if(Mathf.Abs(world.determinant)<1e-10f)throw new Exception("Singular transform");
            var vertices=mesh.vertices;
            var normals=mesh.normals;
            if(normals.Length!=vertices.Length)throw new Exception("Mesh must have vertex normals");
            var nm=world.inverse.transpose;
            var sb=new StringBuilder();
            foreach(var v in vertices)
            {
                var p=ConvertPoint(world.MultiplyPoint3x4(v));
                sb.AppendLine("v "+F(p.x)+" "+F(p.y)+" "+F(p.z));
            }
            foreach(var n in normals)
            {
                var p=ConvertPoint(nm.MultiplyVector(n)).normalized;
                if(p.sqrMagnitude<0.5f)throw new Exception("Invalid normal");
                sb.AppendLine("vn "+F(p.x)+" "+F(p.y)+" "+F(p.z));
            }
            var t=mesh.GetTriangles(sub);
            bool flip=world.determinant>0;
            // axis swap has determinant -1
            for(int i=0;
            i<t.Length;
            i+=3)
            {
                int a=t[i]+1,b=t[i+(flip?2:1)]+1,c=t[i+(flip?1:2)]+1;
                sb.AppendLine("f "+a+"//"+a+" "+b+"//"+b+" "+c+"//"+c);
            }
            File.WriteAllText(path,sb.ToString(),new UTF8Encoding(false));
        }
        public static void Export(GameObject[] roots,Camera camera,string directory)
        {
            if(Directory.Exists(directory))throw new Exception("Output already exists; choose a new directory");
            var errors=new List<string>();
            var renderers=roots.SelectMany(r=>r.GetComponentsInChildren<MeshRenderer>()).Distinct().Where(r=>r.enabled&&r.gameObject.activeInHierarchy).ToArray();
            if(renderers.Length==0)errors.Add("No enabled static MeshRenderers selected");
            if(camera==null||camera.orthographic||camera.usePhysicalProperties||Mathf.Abs(camera.aspect-1)>1e-4f||camera.lensShift!=Vector2.zero||camera.fieldOfView<=0||camera.fieldOfView>=179||float.IsNaN(camera.fieldOfView))errors.Add("Require square perspective camera, no physical camera/lens shift");
            foreach(var root in roots)
            {
                if(root.GetComponentsInChildren<SkinnedMeshRenderer>().Length>0)errors.Add("Skinned meshes unsupported");
                if(root.GetComponentsInChildren<ParticleSystem>().Length>0)errors.Add("Particles unsupported");
                if(root.GetComponentsInChildren<Light>().Any(l=>l.enabled))errors.Add("Unity Lights unsupported; use white emissive mesh");
            }
            var entries=new List<Tuple<MeshRenderer,Mesh,int,string>>();
            foreach(var r in renderers)try
            {
                if(r.HasPropertyBlock())throw new Exception("MaterialPropertyBlock unsupported");
                var mf=r.GetComponent<MeshFilter>();
                if(mf==null||mf.sharedMesh==null)throw new Exception("Missing mesh");
                if(r.sharedMaterials.Length!=mf.sharedMesh.subMeshCount)throw new Exception("Submesh/material count mismatch");
                for(int s=0;
                s<mf.sharedMesh.subMeshCount;
                s++)entries.Add(Tuple.Create(r,mf.sharedMesh,s,MaterialJson(r.sharedMaterials[s])));
            }
            catch(Exception ex)
            {
                errors.Add(r.name+": "+ex.Message);
            }
            if(errors.Count>0)throw new Exception(string.Join("\n",errors));
            Directory.CreateDirectory(directory);
            var objects=new List<string>();
            int j=0;
            foreach(var x in entries)
            {
                string name="mesh_"+(j++).ToString("D4");
                WriteMesh(x.Item2,x.Item3,x.Item1.localToWorldMatrix,Path.Combine(directory,name+".obj"));
                objects.Add(Q(name)+":{\"mesh_path\":"+Q(name+".obj")+",\"material\":"+x.Item4+",\"transform\":{\"translation\":[0,0,0],\"rotation\":[0,0,0],\"scale\":[1,1,1],\"normalize\":false},\"remesh\":false}");
            }
            var cp=ConvertPoint(camera.transform.position);
            var target=ConvertPoint(camera.transform.position+camera.transform.forward);
            var up=ConvertPoint(camera.transform.up);
            string json="{\"scene_name\":\"Unity controlled static export\",\"version\":\"1.0\",\"objects\":{"+string.Join(",",objects)+"},\"cameras\":[{\"position\":"+V(cp)+",\"look_at\":"+V(target)+",\"up\":"+V(up)+",\"fov\":"+F(camera.fieldOfView)+"}]}";
            File.WriteAllText(Path.Combine(directory,"scene.json"),json,new UTF8Encoding(false));
            File.WriteAllText(Path.Combine(directory,"EXPORT-NOTES.txt"),"Coordinate map Unity(x,y,z) -> experiment(x,z,y). World transforms baked; normals inverse transpose; parity-aware face reversal. Square vertical FOV. Materials are controlled approximation: diffuse=baseColor*(1-specular), roughness=1-smoothness, not exact Unity appearance equivalence. Official converter may rebuild normals; compare prediction and Cycles on identical converted mesh. Unity native Lights, environment, baked lighting, postprocess and custom shaders are not represented. Selected roots only. Color values use raw Material.GetColor numeric values without an implicit gamma transform; project color space="+QualitySettings.activeColorSpace+".\n");
        }
        [MenuItem("Tools/RenderFormer/Export Selected Static Roots")]
        static void Menu()
        {
            var camera=Camera.main;
            if(camera==null)throw new Exception("Tag the intended camera MainCamera");
            string parent=EditorUtility.OpenFolderPanel("Choose parent for a NEW export folder","","");
            if(parent.Length==0)return;
            Export(Selection.gameObjects,camera,Path.Combine(parent,"renderformer-"+DateTime.Now.ToString("yyyyMMdd-HHmmss")));
        }
    }
}
#endif
