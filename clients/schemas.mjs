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
export const profileSchema = z
  .object({
    subject: z.literal("cat"),
    up_axis: z.union([z.literal(0), z.literal(1), z.literal(2)]),
    depth_axis: z.union([z.literal(0), z.literal(1), z.literal(2)]),
    regions: z.object({ head: region, body: region }).strict(),
  })
  .strict()
  .refine(
    (p) =>
      p.up_axis !== p.depth_axis &&
      Object.values(p.regions).every(
        (r) =>
          r.bounds.every(([a, b]) => a < b) &&
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
