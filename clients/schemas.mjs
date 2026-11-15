import { z } from "zod";
const finite = z.number().finite();
const bounds = z.tuple([
  z.tuple([finite, finite]),
  z.tuple([finite, finite]),
  z.tuple([finite, finite]),
]);
const region = z
  .object({
    bounds,
    levels: z.array(finite).min(3).max(8),
    min_depth_ratio: finite.gt(0).max(1),
  })
  .strict();
const axis = z.union([z.literal(0), z.literal(1), z.literal(2)]);
const axes = { up_axis: axis, depth_axis: axis };
const partName = z.string().regex(
  /^(?!(?:__proto__|prototype|constructor)$)[^\x00-\x20\x7f-\x9f]{1,48}$/u,
);
// Zod's record parser drops this own key; reject it on the unparsed object too.
const checkedRegionKeys = z.unknown().refine(
  (value) => !(value && typeof value === "object" && Object.hasOwn(value, "__proto__")),
);
const namedRegions = z
  .intersection(checkedRegionKeys, z.record(partName, region))
  .refine((value) => Object.keys(value).length >= 1 && Object.keys(value).length <= 6)
  .meta({ minProperties: 1, maxProperties: 6 });
export const profileSchema = z
  .discriminatedUnion("subject", [
    z.object({
      subject: z.literal("cat"),
      ...axes,
      regions: z.object({ head: region, body: region }).strict(),
    }).strict(),
    z.object({
      subject: z.literal("measured-parts"),
      ...axes,
      regions: namedRegions,
    }).strict(),
  ])
  .refine(
    (p) =>
      p.up_axis !== p.depth_axis &&
      Object.values(p.regions).every(
        (r) =>
          r.bounds.every(([a, b]) => a < b) &&
          (p.subject === "cat" || new Set(r.levels).size === r.levels.length) &&
          r.levels.every(
            (v) => v > r.bounds[p.up_axis][0] && v < r.bounds[p.up_axis][1],
          ),
      ),
    { message: "Invalid axes, bounds or section levels" },
  );
const reviewText = z
  .string()
  .max(4000)
  .refine((x) => x.trim().length > 0);
export const reviewSchema = z
  .object({
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
    decision: z.enum(["pass", "fail", "review"]),
    reference_fidelity: reviewText,
    major_part_volume: reviewText,
    attachment_material: reviewText,
    inspected_views: z.tuple([z.literal("material"), z.literal("clay")]),
  })
  .strict();
